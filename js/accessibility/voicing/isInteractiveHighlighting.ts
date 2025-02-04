// Copyright 2025, University of Colorado Boulder

/**
 * Checks whether something is of the InteractiveHighlightingNodeType type.
 * Extracted to reduce circular dependencies.
 *
 * @author Jesse Greenberg
 */

import type IntentionalAny from '../../../../phet-core/js/types/IntentionalAny.js';
import type { InteractiveHighlightingNodeType } from './InteractiveHighlighting.js';

export function isInteractiveHighlighting( something: IntentionalAny ): something is InteractiveHighlightingNodeType {
  return typeof something === 'object' && ( something as InteractiveHighlightingNodeType )._isInteractiveHighlighting;
}