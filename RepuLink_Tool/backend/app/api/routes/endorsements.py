import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app import crud
from app.api.deps import CurrentUser, SessionDep
from app.models import EndorsementCreate, EndorsementPublic, EndorsementWithUser, User

router = APIRouter(prefix="/endorsements", tags=["endorsements"])


@router.post("/", response_model=EndorsementPublic)
def create_or_update_endorsement(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    endorsement_in: EndorsementCreate,
) -> Any:
    """
    Create or update an endorsement.
    """
    if endorsement_in.endorsed_id == current_user.id:
        raise HTTPException(
            status_code=400,
            detail="You cannot endorse yourself",
        )

    # Check if endorsed user exists
    endorsed_user = session.get(User, endorsement_in.endorsed_id)
    if not endorsed_user:
        raise HTTPException(
            status_code=404,
            detail="User to endorse not found",
        )

    try:
        endorsement = crud.create_or_update_endorsement(
            session=session,
            endorser_id=current_user.id,
            endorsed_id=endorsement_in.endorsed_id,
            confidence=endorsement_in.confidence,
        )
        return endorsement
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/endorsed-by-me", response_model=list[EndorsementWithUser])
def get_my_endorsements(
    *,
    session: SessionDep,
    current_user: CurrentUser,
) -> Any:
    """
    Get all endorsements made by the current user.
    """
    endorsements = crud.get_endorsements_with_user_info(
        session=session, endorser_id=current_user.id
    )
    return endorsements


@router.get("/endorsing-me", response_model=list[EndorsementWithUser])
def get_my_endorsers(
    *,
    session: SessionDep,
    current_user: CurrentUser,
) -> Any:
    """
    Get all users endorsing the current user.
    """
    endorsements = crud.get_endorsers_with_user_info(
        session=session, endorsed_id=current_user.id
    )
    return endorsements


@router.get("/{user_id}/endorsed-by", response_model=list[EndorsementWithUser])
def get_user_endorsements(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    user_id: uuid.UUID,
) -> Any:
    """
    Get all endorsements made by a specific user.
    """
    endorsements = crud.get_endorsements_with_user_info(
        session=session, endorser_id=user_id
    )
    return endorsements


@router.get("/{user_id}/endorsers", response_model=list[EndorsementWithUser])
def get_user_endorsers(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    user_id: uuid.UUID,
) -> Any:
    """
    Get all endorsers of a specific user.
    """
    endorsements = crud.get_endorsers_with_user_info(
        session=session, endorsed_id=user_id
    )
    return endorsements
