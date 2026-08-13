"""Классы объектов проекта: то, что пользователь ищет на своих фотографиях.

Имя класса — это текстовый запрос к движку детекции (OWLv2/LLMDet) и одновременно
Annotation.label. Внешнего ключа из аннотаций сюда нет: удаление класса не должно
превращать уже размеченные боксы в мусор без явного согласия (см. DELETE ?force).
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models import Annotation, Document, Image, Project, ProjectClass
from app.schemas import (
    ProjectClassCreate,
    ProjectClassOut,
    ProjectClassReplace,
    ProjectClassUpdate,
)

router = APIRouter(tags=["classes"])

# Различимые цвета боксов (tailwind-палитра), назначаются по кругу: цвет класса
# должен быть предсказуем, случайные оттенки сливаются друг с другом.
PALETTE: tuple[str, ...] = (
    "#3b82f6",  # blue
    "#ef4444",  # red
    "#22c55e",  # green
    "#f59e0b",  # amber
    "#a855f7",  # purple
    "#06b6d4",  # cyan
    "#ec4899",  # pink
    "#84cc16",  # lime
    "#f97316",  # orange
    "#14b8a6",  # teal
    "#6366f1",  # indigo
    "#eab308",  # yellow
)


def palette_color(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


async def get_project_or_404(session: AsyncSession, project_id: uuid.UUID) -> Project:
    project = await session.get(Project, project_id)
    if project is None:
        raise HTTPException(404, "Project not found")
    return project


async def get_class_or_404(
    session: AsyncSession, class_id: uuid.UUID
) -> ProjectClass:
    item = await session.get(ProjectClass, class_id)
    if item is None:
        raise HTTPException(404, "Class not found")
    return item


async def list_classes(
    session: AsyncSession, project_id: uuid.UUID
) -> list[ProjectClass]:
    result = await session.execute(
        select(ProjectClass)
        .where(ProjectClass.project_id == project_id)
        .order_by(ProjectClass.sort_order, ProjectClass.created_at)
    )
    return list(result.scalars().all())


def project_images(project_id: uuid.UUID):
    return (
        select(Image.id)
        .join(Document, Image.document_id == Document.id)
        .where(Document.project_id == project_id)
    )


async def annotations_count(
    session: AsyncSession, project_id: uuid.UUID, name: str
) -> int:
    """Сколько аннотаций проекта размечено этим классом."""
    result = await session.execute(
        select(func.count())
        .select_from(Annotation)
        .where(
            Annotation.image_id.in_(project_images(project_id)),
            Annotation.label == name,
        )
    )
    return int(result.scalar_one())


async def delete_annotations_for_class(
    session: AsyncSession, project_id: uuid.UUID, name: str
) -> None:
    await session.execute(
        delete(Annotation).where(
            Annotation.image_id.in_(project_images(project_id)),
            Annotation.label == name,
        )
    )


async def rename_annotations(
    session: AsyncSession, project_id: uuid.UUID, old: str, new: str
) -> None:
    """Аннотации ссылаются на класс по имени — переименование тянет их за собой."""
    await session.execute(
        update(Annotation)
        .where(
            Annotation.image_id.in_(project_images(project_id)),
            Annotation.label == old,
        )
        .values(label=new)
    )


def ensure_name_free(
    existing: list[ProjectClass], name: str, skip_id: uuid.UUID | None = None
) -> None:
    """Имя уникально в проекте. Сравнение без учёта регистра: «Sofa» и «sofa» —
    один и тот же запрос к движку, два таких класса дали бы дубли боксов."""
    lowered = name.casefold()
    for item in existing:
        if item.id != skip_id and item.name.casefold() == lowered:
            raise HTTPException(409, f"Класс {item.name!r} уже есть в проекте")


@router.get("/projects/{project_id}/classes", response_model=list[ProjectClassOut])
async def get_classes(
    project_id: uuid.UUID, session: AsyncSession = Depends(get_session)
):
    await get_project_or_404(session, project_id)
    return await list_classes(session, project_id)


@router.post(
    "/projects/{project_id}/classes", response_model=ProjectClassOut, status_code=201
)
async def create_class(
    project_id: uuid.UUID,
    body: ProjectClassCreate,
    session: AsyncSession = Depends(get_session),
):
    await get_project_or_404(session, project_id)
    existing = await list_classes(session, project_id)
    ensure_name_free(existing, body.name)

    sort_order = max((c.sort_order for c in existing), default=-1) + 1
    item = ProjectClass(
        project_id=project_id,
        name=body.name,
        color=body.color or palette_color(sort_order),
        sort_order=sort_order,
    )
    session.add(item)
    try:
        await session.commit()
    except IntegrityError:  # два одинаковых POST подряд — тоже конфликт, не 500
        await session.rollback()
        raise HTTPException(409, f"Класс {body.name!r} уже есть в проекте")
    await session.refresh(item)
    return item


@router.put("/projects/{project_id}/classes", response_model=list[ProjectClassOut])
async def replace_classes(
    project_id: uuid.UUID,
    body: ProjectClassReplace,
    force: bool = Query(False),
    session: AsyncSession = Depends(get_session),
):
    """Заменить список классов целиком (главный сценарий — настройка проекта).

    Уцелевшие классы сохраняют id и цвет, чтобы боксы в Review UI не перекрасились.
    Класс, на котором уже есть аннотации, молча не выбрасываем — 409, как и в DELETE.
    """
    await get_project_or_404(session, project_id)
    existing = await list_classes(session, project_id)

    # дубли в присланном списке — почти всегда опечатка в textarea, не ошибка
    wanted: list[str] = []
    seen: set[str] = set()
    for name in body.names:
        if name.casefold() not in seen:
            seen.add(name.casefold())
            wanted.append(name)

    by_name = {c.name.casefold(): c for c in existing}
    removed = [c for c in existing if c.name.casefold() not in seen]
    in_use = []
    in_use_total = 0
    for item in removed:
        used = await annotations_count(session, project_id, item.name)
        if used:
            in_use.append(f"{item.name} ({used})")
            in_use_total += used
    if in_use and not force:
        # тело той же формы, что у DELETE: число первым, чтобы фронт его достал
        raise HTTPException(
            409,
            {
                "annotations": in_use_total,
                "message": "Нельзя убрать классы с уже размеченными аннотациями: "
                + ", ".join(in_use)
                + ". Повторите с ?force=true, чтобы удалить их вместе с разметкой.",
                "classes": [c.name for c in removed],
            },
        )

    for item in removed:
        if force:
            await delete_annotations_for_class(session, project_id, item.name)
        await session.delete(item)

    # цвет новому классу — первый свободный в палитре: уцелевшие сохраняют свой,
    # и назначение по индексу дало бы двум классам один цвет
    used_colors = {c.color.lower() for c in existing if c.name.casefold() in seen}

    result: list[ProjectClass] = []
    for order, name in enumerate(wanted):
        item = by_name.get(name.casefold())
        if item is None:
            color = next(
                (c for c in PALETTE if c.lower() not in used_colors),
                palette_color(order),
            )
            used_colors.add(color.lower())
            item = ProjectClass(
                project_id=project_id,
                name=name,
                color=color,
                sort_order=order,
            )
            session.add(item)
        else:
            if item.name != name:  # поменялся регистр — тянем аннотации за собой
                await rename_annotations(session, project_id, item.name, name)
                item.name = name
            item.sort_order = order
        result.append(item)

    await session.commit()
    return result


@router.patch("/classes/{class_id}", response_model=ProjectClassOut)
async def update_class(
    class_id: uuid.UUID,
    body: ProjectClassUpdate,
    session: AsyncSession = Depends(get_session),
):
    item = await get_class_or_404(session, class_id)
    updates = body.model_dump(exclude_unset=True, exclude_none=True)

    new_name = updates.get("name")
    if new_name is not None and new_name != item.name:
        existing = await list_classes(session, item.project_id)
        ensure_name_free(existing, new_name, skip_id=item.id)
        await rename_annotations(session, item.project_id, item.name, new_name)

    for field, value in updates.items():
        setattr(item, field, value)

    await session.commit()
    await session.refresh(item)
    return item


@router.delete("/classes/{class_id}", status_code=204)
async def delete_class(
    class_id: uuid.UUID,
    force: bool = Query(False),
    session: AsyncSession = Depends(get_session),
):
    item = await get_class_or_404(session, class_id)
    used = await annotations_count(session, item.project_id, item.name)
    if used and not force:
        # число — первым полем: UI достаёт его из тела ответа
        raise HTTPException(
            409,
            {
                "annotations": used,
                "message": (
                    f"Класс {item.name!r} используется в аннотациях: {used}. "
                    "Повторите с ?force=true, чтобы удалить их вместе с классом"
                ),
            },
        )
    if used:
        await session.execute(
            delete(Annotation).where(
                Annotation.image_id.in_(project_images(item.project_id)),
                Annotation.label == item.name,
            )
        )
    await session.delete(item)
    await session.commit()
