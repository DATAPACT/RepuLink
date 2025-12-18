from typing import Any, List
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import select

from app.api.deps import CurrentUser, SessionDep
from app import crud
from app.models import (
    InteractionCreate,
    InteractionPublic,
    Interaction,
    Message,
    User,
    RatingCreate,
    RatingPublic,
)

router = APIRouter(prefix="/interactions", tags=["interactions"])


@router.post("/", response_model=InteractionPublic)
def create_interaction(
    *, body: InteractionCreate, session: SessionDep, current_user: CurrentUser
) -> Any:
    """Create a new interaction request from current user to a target user."""
    target = session.get(User, body.target_id)
    if not target:
        raise HTTPException(status_code=404, detail="Target user not found")
    interaction = crud.create_interaction(
        session=session, initiator_id=current_user.id, target_id=body.target_id, message=body.message
    )
    return interaction


class RespondBody(Message):
    # reuse Message.message as a short message if desired
    pass


@router.post("/{interaction_id}/respond", response_model=Message)
def respond_interaction(
    interaction_id: uuid.UUID,
    accept: bool,
    session: SessionDep,
    current_user: CurrentUser,
) -> Any:
    """Accept or deny an interaction. Only the target user can respond."""
    try:
        interaction = crud.respond_interaction(
            session=session, interaction_id=interaction_id, responder_id=current_user.id, accept=accept
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return Message(message=f"Interaction {interaction.status}")


@router.get("/users/{user_id}", response_model=List[InteractionPublic])
def list_user_interactions(
    user_id: uuid.UUID,
    session: SessionDep,
    current_user: CurrentUser,
    role: str | None = Query(None),
    skip: int = Query(0),
    limit: int = Query(100),
) -> Any:
    """List interactions for a user. Only the user themselves or superuser can list."""
    if user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not authorized")
    interactions = crud.get_user_interactions(
        session=session, user_id=user_id, role=role, skip=skip, limit=limit
    )
    return interactions


@router.get("/{interaction_id}/ratings")
def list_ratings_for_interaction(
    interaction_id: uuid.UUID,
    session: SessionDep,
    current_user: CurrentUser,
) -> Any:
    """List ratings for an interaction with rater information. Only participants or superuser can view."""
    interaction = session.get(Interaction, interaction_id)
    if not interaction:
        raise HTTPException(status_code=404, detail="Interaction not found")
    if (
        current_user.id not in (interaction.initiator_id, interaction.target_id)
        and not current_user.is_superuser
    ):
        raise HTTPException(status_code=403, detail="Not authorized")
    ratings = crud.get_ratings_for_interaction(session=session, interaction_id=interaction_id)
    
    # Enrich with rater info
    result = []
    for rating in ratings:
        rater = session.get(User, rating.rater_id)
        result.append({
            "id": str(rating.id),
            "interaction_id": str(rating.interaction_id),
            "rater_id": str(rating.rater_id),
            "rating": rating.rating,
            "comment": rating.comment,
            "created_at": rating.created_at.isoformat(),
            "rater_email": rater.email if rater else "Unknown",
            "rater_full_name": rater.full_name if rater else "Unknown User",
        })
    return result



@router.post("/{interaction_id}/rating", response_model=RatingPublic)
def create_rating_for_interaction(
    interaction_id: uuid.UUID,
    body: RatingCreate,
    session: SessionDep,
    current_user: CurrentUser,
) -> Any:
    """Create a rating for an accepted interaction. Only participants can rate."""
    try:
        rating = crud.add_rating(
            session=session,
            interaction_id=interaction_id,
            rater_id=current_user.id,
            rating=body.rating,
            comment=body.comment,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return rating
