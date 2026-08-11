from pydantic import BaseModel, Field
class VendorCategoryCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
class VendorCategoryUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
