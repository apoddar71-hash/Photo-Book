"""Data models and Pydantic schemas for the Photo Book Builder."""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Normalized bounding box [0..1000] in Gemini format."""

    ymin: int = Field(ge=0, le=1000, description="Top Y coordinate (0-1000)")
    xmin: int = Field(ge=0, le=1000, description="Left X coordinate (0-1000)")
    ymax: int = Field(ge=0, le=1000, description="Bottom Y coordinate (0-1000)")
    xmax: int = Field(ge=0, le=1000, description="Right X coordinate (0-1000)")
    label: str = Field(default="face", description="Object label, e.g. 'face' or 'subject'")

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Return (left, top, right, bottom) in image pixel coordinates."""
        y0, y1 = min(self.ymin, self.ymax), max(self.ymin, self.ymax)
        x0, x1 = min(self.xmin, self.xmax), max(self.xmin, self.xmax)
        left = max(0, min(width, int(round(x0 / 1000.0 * width))))
        top = max(0, min(height, int(round(y0 / 1000.0 * height))))
        right = max(left + 1, min(width, int(round(x1 / 1000.0 * width))))
        bottom = max(top + 1, min(height, int(round(y1 / 1000.0 * height))))
        return left, top, right, bottom


class GeminiPhotoAnalysis(BaseModel):
    """Structured output schema returned by Gemini 3.8 Flash per photo."""

    is_document_or_screenshot: bool = Field(
        default=False,
        description="True if the image is a photo of a paper document, receipt, invoice, ticket, menu, text slide, or screenshot rather than a personal photo.",
    )
    document_reason: Optional[str] = Field(
        default=None,
        description="If is_document_or_screenshot is True, brief explanation (e.g. 'Hotel invoice document', 'Restaurant receipt').",
    )
    is_blurry_or_flawed: bool = Field(
        default=False,
        description="True if the photo is severely blurry, accidental pocket shot, or unusable.",
    )
    has_sideways_or_inverted_orientation: bool = Field(
        default=False,
        description="True if a person's face/head or the horizon/scene is oriented sideways (rotated 90 degrees) or upside-down.",
    )
    aesthetic_score: float = Field(
        default=7.5,
        ge=1.0,
        le=10.0,
        description="Overall composition, lighting, and visual appeal score from 1 to 10.",
    )
    sharpness_score: float = Field(
        default=8.0,
        ge=1.0,
        le=10.0,
        description="Focus and technical clarity score from 1 to 10.",
    )
    emotional_score: float = Field(
        default=7.5,
        ge=1.0,
        le=10.0,
        description="Storytelling warmth, expressions, and memory value from 1 to 10.",
    )
    cover_worthiness_score: float = Field(
        default=7.0,
        ge=1.0,
        le=10.0,
        description="Suitability as the 10x10 front cover hero image from 1 to 10.",
    )
    subject_summary: str = Field(
        default="",
        description="Detailed visual description of subjects, setting, colors, and action for similarity detection.",
    )
    location_or_landmark: str = Field(
        default="",
        description="Detected landmark, setting, or place name.",
    )
    event_theme: str = Field(
        default="",
        description="Short cluster theme label describing the activity or setting.",
    )
    caption: str = Field(
        default="",
        description="Warm, engaging photo book caption (strictly 1 to 2 lines max, under 140 characters).",
    )
    people_count: int = Field(
        default=0,
        ge=0,
        description="Number of people visible in the photo.",
    )
    faces_boxes: list[BoundingBox] = Field(
        default_factory=list,
        description="Bounding boxes [0..1000] for every visible human head/face (include full head from crown of hair to chin) so cropping never cuts off a face.",
    )
    salient_subject_box: BoundingBox = Field(
        default_factory=lambda: BoundingBox(ymin=150, xmin=150, ymax=850, xmax=850, label="subject"),
        description="Bounding box [0..1000] enclosing the main subject(s) of interest for aesthetic framing.",
    )

    @property
    def composite_score(self) -> float:
        return round(
            0.45 * self.aesthetic_score
            + 0.30 * self.emotional_score
            + 0.25 * self.sharpness_score,
            2,
        )


class BatchPhotoAnalysisItem(GeminiPhotoAnalysis):
    """Single photo analysis entry inside a multi-photo batch response."""

    photo_id: str = Field(description="Exact photo_id matching the input photo in the batch.")


class BatchGeminiPhotoAnalysis(BaseModel):
    """Structured output schema for multi-photo batch vision analysis."""

    items: list[BatchPhotoAnalysisItem] = Field(
        default_factory=list,
        description="Analysis results for each photo in the batch, matched by photo_id.",
    )


class ClusterRefinementItem(BaseModel):
    """Refined chapter title and 1-2 line group caption for a photo cluster."""

    cluster_id: str = Field(description="Exact cluster_id matching the input cluster.")
    title: str = Field(description="Evocative chapter title for this chronological group of photos.")
    group_caption: str = Field(description="Warm 1 to 2 line max summary caption for this group of photos.")


class BookCurationRefinement(BaseModel):
    """Fast Stage-2 curation response from the text/curation model."""

    book_title: str = Field(description="Cohesive, memorable title for the 10x10 photo book.")
    clusters: list[ClusterRefinementItem] = Field(default_factory=list)


PhotoStatus = Literal[
    "selected",
    "excluded_document",
    "excluded_duplicate",
    "excluded_similar",
    "excluded_low_quality",
]


class PhotoMetadata(BaseModel):
    """Complete record for a scanned photo in the input folder."""

    photo_id: str
    filename: str
    file_path: str
    original_width: int
    original_height: int
    analysis_width: int
    analysis_height: int
    analysis_path: str
    captured_at: str
    captured_timestamp: float
    camera_model: Optional[str] = None
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    dhash: str
    ahash: str
    color_signature: list[float] = Field(default_factory=list)
    status: PhotoStatus = "selected"
    exclusion_reason: Optional[str] = None
    duplicate_of: Optional[str] = None
    cluster_id: Optional[str] = None
    analysis: GeminiPhotoAnalysis = Field(default_factory=GeminiPhotoAnalysis)

    @property
    def aspect_ratio(self) -> float:
        return self.original_width / max(1, self.original_height)

    @property
    def orientation(self) -> Literal["landscape", "portrait", "square"]:
        ar = self.aspect_ratio
        if ar > 1.15:
            return "landscape"
        if ar < 0.87:
            return "portrait"
        return "square"


class PhotoCluster(BaseModel):
    """A chronological thematic group/chapter of photos."""

    cluster_id: str
    title: str
    date_range_label: str
    group_caption: str = Field(description="1-2 line max caption summarizing this group of photos.")
    photo_ids: list[str] = Field(default_factory=list)


class PhotoSlot(BaseModel):
    """One positioned & cropped photo inside a 10x10 inch page."""

    photo_id: str
    filename: str
    slot_index: int
    x_in: float
    y_in: float
    width_in: float
    height_in: float
    crop_box_px: tuple[int, int, int, int] = Field(
        description="(left, top, right, bottom) in original image pixel coordinates"
    )
    crop_box_normalized: tuple[int, int, int, int] = Field(
        description="(ymin, xmin, ymax, xmax) in 0..1000 coordinates"
    )
    cropped_image_relpath: str
    caption: str
    faces_count: int = 0
    faces_preserved: bool = True


PageTemplateId = Literal[
    "cover_hero",
    "single_full_square",
    "single_editorial",
    "single_panoramic",
    "two_side_by_side",
    "two_stacked_horizontal",
    "two_asymmetric",
    "three_hero_left",
    "three_hero_top",
    "three_columns",
]


class BookPage(BaseModel):
    """A 10x10 inch page in the photo book containing 1, 2, or 3 photos max."""

    page_number: int
    page_type: Literal["cover", "photo_page"] = "photo_page"
    template_id: PageTemplateId
    cluster_id: Optional[str] = None
    chapter_title: Optional[str] = None
    group_caption: Optional[str] = None
    slots: list[PhotoSlot] = Field(default_factory=list)


class PhotoBookResult(BaseModel):
    """Complete output of the Photo Book Builder pipeline."""

    book_id: str
    title: str
    subtitle: str
    folder_path: str
    created_at: str
    model_requested: str
    model_used: str
    curation_model_used: str = "gemini-2.5-flash-lite"
    hero_model_used: str = "gemini-3.8-flash"
    layout_model_used: str = "local_geometric"
    batch_size: int = 6
    build_duration_seconds: float = 0.0
    stage_timings: dict[str, float] = Field(default_factory=dict)
    model_note: Optional[str] = None
    thinking_level: str = "medium"
    media_resolution: str = "high"
    analysis_pixel_range: str = "1,120 to 2,240 px"
    book_size_inches: tuple[float, float] = (10.0, 10.0)
    hero_cover_photo_id: str
    total_scanned: int
    total_selected: int
    total_excluded_duplicates: int
    total_excluded_similar: int
    total_excluded_documents: int
    total_excluded_low_quality: int
    clusters: list[PhotoCluster]
    pages: list[BookPage]
    photos: list[PhotoMetadata]
    output_dir: str
    analysis_json_path: str
    analysis_markdown_path: str
    html_preview_path: str
    pdf_path: str
