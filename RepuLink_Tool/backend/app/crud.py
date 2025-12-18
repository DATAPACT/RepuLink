import uuid
from typing import Any

from sqlmodel import Session, select
from sqlalchemy import or_

from app.core.security import get_password_hash, verify_password
from app.models import Item, ItemCreate, User, UserCreate, UserUpdate


def create_user(*, session: Session, user_create: UserCreate) -> User:
    db_obj = User.model_validate(
        user_create, update={"hashed_password": get_password_hash(user_create.password)}
    )
    session.add(db_obj)
    session.commit()
    session.refresh(db_obj)
    return db_obj


def update_user(*, session: Session, db_user: User, user_in: UserUpdate) -> Any:
    user_data = user_in.model_dump(exclude_unset=True)
    extra_data = {}
    if "password" in user_data:
        password = user_data["password"]
        hashed_password = get_password_hash(password)
        extra_data["hashed_password"] = hashed_password
    db_user.sqlmodel_update(user_data, update=extra_data)
    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return db_user


def get_user_by_email(*, session: Session, email: str) -> User | None:
    statement = select(User).where(User.email == email)
    session_user = session.exec(statement).first()
    return session_user


def search_users(*, session: Session, query: str, limit: int = 10) -> list[User]:
    """Search users by email or full_name (case-insensitive)."""
    statement = select(User).where(
        or_(User.email.ilike(f"%{query}%"), User.full_name.ilike(f"%{query}%"))
    ).limit(limit)
    results = session.exec(statement).all()
    return results


def authenticate(*, session: Session, email: str, password: str) -> User | None:
    db_user = get_user_by_email(session=session, email=email)
    if not db_user:
        return None
    if not verify_password(password, db_user.hashed_password):
        return None
    return db_user


def create_item(*, session: Session, item_in: ItemCreate, owner_id: uuid.UUID) -> Item:
    db_item = Item.model_validate(item_in, update={"owner_id": owner_id})
    session.add(db_item)
    session.commit()
    session.refresh(db_item)
    return db_item


def create_interaction(
    *, session: Session, initiator_id: uuid.UUID, target_id: uuid.UUID, message: str | None = None
) -> "Interaction":
    from app.models import Interaction, InteractionCreate
    from sqlmodel import select

    if initiator_id == target_id:
        raise ValueError("Cannot create an interaction with yourself")

    # prevent duplicate pending interactions from same initiator to same target
    statement = select(Interaction).where(
        (Interaction.initiator_id == initiator_id)
        & (Interaction.target_id == target_id)
        & (Interaction.status == "pending")
    )
    existing = session.exec(statement).first()
    if existing:
        raise ValueError("A pending interaction already exists")

    interaction_in = InteractionCreate.model_validate({"target_id": target_id, "message": message})
    db_obj = Interaction.model_validate(interaction_in, update={"initiator_id": initiator_id})
    session.add(db_obj)
    session.commit()
    session.refresh(db_obj)
    return db_obj


def respond_interaction(
    *, session: Session, interaction_id: uuid.UUID, responder_id: uuid.UUID, accept: bool
) -> "Interaction":
    from app.models import Interaction

    db_obj = session.get(Interaction, interaction_id)
    if not db_obj:
        raise ValueError("Interaction not found")
    # only target can respond
    if db_obj.target_id != responder_id:
        raise ValueError("Not authorized to respond to this interaction")
    db_obj.status = "accepted" if accept else "denied"
    db_obj.updated_at = __import__("datetime").datetime.utcnow()
    session.add(db_obj)
    session.commit()
    session.refresh(db_obj)
    return db_obj


def get_user_interactions(
    *, session: Session, user_id: uuid.UUID, role: str | None = None, skip: int = 0, limit: int = 100
) -> list["Interaction"]:
    from app.models import Interaction
    from sqlmodel import select

    statement = select(Interaction)
    if role == "initiator":
        statement = statement.where(Interaction.initiator_id == user_id)
    elif role == "target":
        statement = statement.where(Interaction.target_id == user_id)
    else:
        statement = statement.where((Interaction.initiator_id == user_id) | (Interaction.target_id == user_id))
    statement = statement.offset(skip).limit(limit)
    results = session.exec(statement).all()
    return results


def add_rating(*, session: Session, interaction_id: uuid.UUID, rater_id: uuid.UUID, rating: int, comment: str | None = None) -> "Rating":
    from app.models import Interaction, Rating, RatingCreate
    from sqlmodel import select

    interaction = session.get(Interaction, interaction_id)
    if not interaction:
        raise ValueError("Interaction not found")
    if interaction.status != "accepted":
        raise ValueError("Can only rate an accepted interaction")
    # Only initiator or target may rate
    if rater_id not in (interaction.initiator_id, interaction.target_id):
        raise ValueError("Not authorized to rate this interaction")

    # Prevent duplicate rating by same rater for same interaction
    statement = select(Rating).where(
        (Rating.interaction_id == interaction_id) & (Rating.rater_id == rater_id)
    )
    existing = session.exec(statement).first()
    if existing:
        raise ValueError("User has already rated this interaction")

    rating_in = RatingCreate.model_validate({"rating": rating, "comment": comment})
    db_obj = Rating.model_validate(rating_in, update={"interaction_id": interaction_id, "rater_id": rater_id})
    session.add(db_obj)
    session.commit()
    session.refresh(db_obj)
    return db_obj


def get_ratings_for_interaction(*, session: Session, interaction_id: uuid.UUID) -> list["Rating"]:
    from app.models import Rating
    from sqlmodel import select

    statement = select(Rating).where(Rating.interaction_id == interaction_id)
    return session.exec(statement).all()


def create_or_update_endorsement(
    *, session: Session, endorser_id: uuid.UUID, endorsed_id: uuid.UUID, confidence: float
) -> "Endorsement":
    from app.models import Endorsement, EndorsementCreate

    if endorser_id == endorsed_id:
        raise ValueError("Cannot endorse yourself")

    statement = select(Endorsement).where(
        (Endorsement.endorser_id == endorser_id)
        & (Endorsement.endorsed_id == endorsed_id)
    )
    existing = session.exec(statement).first()

    if existing:
        existing.confidence = confidence
        existing.updated_at = __import__("datetime").datetime.utcnow()
        session.add(existing)
        session.commit()
        session.refresh(existing)
        return existing
    else:
        endorsement_in = EndorsementCreate(endorsed_id=endorsed_id, confidence=confidence)
        db_obj = Endorsement.model_validate(endorsement_in, update={"endorser_id": endorser_id})
        session.add(db_obj)
        session.commit()
        session.refresh(db_obj)
        return db_obj


def get_endorsements_by_user(*, session: Session, endorser_id: uuid.UUID) -> list["Endorsement"]:
    from app.models import Endorsement

    statement = select(Endorsement).where(Endorsement.endorser_id == endorser_id)
    return session.exec(statement).all()


def get_endorsers_for_user(*, session: Session, endorsed_id: uuid.UUID) -> list["Endorsement"]:
    from app.models import Endorsement

    statement = select(Endorsement).where(Endorsement.endorsed_id == endorsed_id)
    return session.exec(statement).all()


def get_endorsements_with_user_info(
    *, session: Session, endorser_id: uuid.UUID
) -> list[dict[str, Any]]:
    """Get all endorsements by a user with endorsed user info"""
    from app.models import Endorsement, User

    statement = (
        select(Endorsement, User.email, User.full_name)
        .join(User, Endorsement.endorsed_id == User.id)
        .where(Endorsement.endorser_id == endorser_id)
    )
    results = session.exec(statement).all()
    return [
        {
            "id": e.id,
            "endorser_id": e.endorser_id,
            "endorsed_id": e.endorsed_id,
            "confidence": e.confidence,
            "created_at": e.created_at,
            "updated_at": e.updated_at,
            "user_email": email,
            "user_full_name": full_name,
        }
        for e, email, full_name in results
    ]


def get_endorsers_with_user_info(
    *, session: Session, endorsed_id: uuid.UUID
) -> list[dict[str, Any]]:
    """Get all endorsers of a user with endorser user info"""
    from app.models import Endorsement, User

    statement = (
        select(Endorsement, User.email, User.full_name)
        .join(User, Endorsement.endorser_id == User.id)
        .where(Endorsement.endorsed_id == endorsed_id)
    )
    results = session.exec(statement).all()
    return [
        {
            "id": e.id,
            "endorser_id": e.endorser_id,
            "endorsed_id": e.endorsed_id,
            "confidence": e.confidence,
            "created_at": e.created_at,
            "updated_at": e.updated_at,
            "user_email": email,
            "user_full_name": full_name,
        }
        for e, email, full_name in results
    ]

