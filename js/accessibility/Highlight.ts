// Copyright 2025, University of Colorado Boulder

/**
 * Highlights displayed by the overlay support these types. Highlight behavior works like the following:
 * - If value is null, the highlight will use default stylings of HighlightPath and surround the Node with focus.
 * - If value is a Shape the Shape is set to a HighlightPath with default stylings in the global coordinate frame.
 * - If you provide a Node it is your responsibility to position it in the global coordinate frame.
 * - If the value is 'invisible' no highlight will be displayed at all.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import Shape from '../../../kite/js/Shape.js';
import type Node from '../nodes/Node.js';

export type Highlight = Node | Shape | null | 'invisible';