from __future__ import annotations

from PIL import Image

from bulletbot.domain.models import PageObservation, PageState
from bulletbot.vision.layout_calibrator import LayoutCalibrator
from bulletbot.vision.page_classifier import PageClassifier
from bulletbot.vision.page_templates import PageTemplateMatcher


class PageRecognizer:
    """Fuse structural, OCR-anchor, and multi-scale template page evidence."""

    _GENERIC_STATES = {PageState.UNKNOWN, PageState.MARKET_OTHER}

    def __init__(
        self,
        structural: PageClassifier | None = None,
        calibrator: LayoutCalibrator | None = None,
        templates: PageTemplateMatcher | None = None,
    ) -> None:
        self._structural = structural or PageClassifier()
        self._calibrator = calibrator or LayoutCalibrator()
        self._templates = templates or PageTemplateMatcher()

    def recognize(self, image: Image.Image) -> PageObservation:
        structural = self._structural.classify(image)
        templates = self._templates.match(image)
        template_states = {item.state for item in templates}
        template_evidence = tuple(
            f"模板 {item.name} {item.score:.0%}" for item in templates
        )

        claim_matches = {
            item.name: item
            for item in templates
            if item.state is PageState.MARKET_CLAIM_COMPLETE
            and item.score >= 0.84
        }
        claim_anchor_names = {
            "market_claim_complete_title",
            "market_claim_coin",
        }
        if claim_anchor_names.issubset(claim_matches):
            # This passive result page has no persistent navigation chrome.
            # The dedicated title and coin anchors only authorize the
            # reversible Space action that returns to the market sell page.
            return PageObservation(
                PageState.MARKET_CLAIM_COMPLETE,
                min(claim_matches[name].score for name in claim_anchor_names),
                template_evidence,
            )

        if (
            structural.state not in self._GENERIC_STATES
            and structural.confidence >= 0.85
            and structural.state in template_states
        ):
            return PageObservation(
                structural.state,
                min(0.99, structural.confidence + 0.01),
                structural.evidence + template_evidence,
            )

        # DetailOcr's structural check already requires the price action,
        # slider, minus, and plus controls. Avoid another OCR pass when all of
        # those controls have been found.
        if structural.state == PageState.DETAIL and structural.confidence >= 0.95:
            return structural

        layout = self._calibrator.analyze(image)

        if structural.state == layout.state and structural.state != PageState.UNKNOWN:
            state = structural.state
            confidence = min(0.99, max(structural.confidence, layout.confidence) + 0.01)
        elif structural.state in self._GENERIC_STATES and layout.state not in self._GENERIC_STATES:
            state = layout.state
            confidence = layout.confidence
        elif layout.state in self._GENERIC_STATES and structural.state not in self._GENERIC_STATES:
            state = structural.state
            confidence = structural.confidence
        elif structural.state in template_states and layout.state not in template_states:
            state = structural.state
            confidence = structural.confidence
        elif layout.state in template_states and structural.state not in template_states:
            state = layout.state
            confidence = layout.confidence
        elif structural.state == PageState.UNKNOWN and layout.state == PageState.UNKNOWN:
            state = PageState.UNKNOWN
            confidence = 0.0
        else:
            state = PageState.UNKNOWN
            confidence = 0.0

        evidence = list(structural.evidence)
        evidence.extend(value for value in layout.evidence if value not in evidence)
        evidence.extend(template_evidence)
        if state == PageState.UNKNOWN and structural.state != layout.state:
            evidence.append(
                f"页面证据冲突：结构 {structural.state.value}，文字/控件 {layout.state.value}"
            )
        return PageObservation(state, confidence, tuple(evidence))
