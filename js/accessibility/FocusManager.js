// Copyright 2021, University of Colorado Boulder

/**
 * Manages the Properties which signify and control where various forms of focus are. A Focus
 * just contains the Trail pointing to the Node with focus and a Display whose root is at the
 * root of that Trail. So it can be used for more than just DOM focus. At the time of this writing,
 * the forms of Focus include
 *
 *  - DOM Focus - The Focus Trail points to the Node whose element has DOM focus in the Parallel DOM.
 *                Only one element can have focus at a time (DOM limitation) so this is managed by a static on
 *                FocusManager.
 *  - Pointer focus - The Focus trail points to a Node that supports Highlighting with pointer events.
 *  - Reading Block Focus - The Focus Trail points to a Node that supports ReadingBlocks, and is active
 *                          while the ReadingBlock content is being spoken for Voicing. See ReadingBlock.js
 *
 * There may be other forms of Focus in the future.
 *
 * This class also has a few Properties that control the behavior of the Display's HighlightOverlay.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import BooleanProperty from '../../../axon/js/BooleanProperty.js';
import Property from '../../../axon/js/Property.js';
import Tandem from '../../../tandem/js/Tandem.js';
import NullableIO from '../../../tandem/js/types/NullableIO.js';
import scenery from '../scenery.js';
import Focus from './Focus.js';

class FocusManager {
  constructor() {

    // @public {Property.<Focus|null>} - This Property whose Focus Trail points to the Node under the pointer to
    // support features of Voicing and Interactive Highlights. Nodes that compose MouseHighlighting can
    // receive this Focus and a highlight may appear around it.
    this.pointerFocusProperty = new Property( null );

    // @public {Property.<Focus|null> - The Property that indicates which Node that uses ReadingBlock is currently
    // active. Used by the HighlightOverlay to highlight ReadingBlock Nodes whose content is being spoken.
    this.readingBlockFocusProperty = new Property( null );

    // @public {BooleanProperty} - A Property that controls whether the highlight for the pointerFocusProperty
    // is locked so that the HighlightOverlay will wait to update the highlight for the pointerFocusProperty. This
    // is useful when the pointer has begun to interact with a Node that uses MouseHighlighting, but the mouse
    // has moved over another during interaction. The highlight should remain on the Node receiving interaction
    // and wait to update until interaction completes.
    this.pointerFocusLockedProperty = new Property( false );

    // @public - Controls whether or not highlights related to PDOM focus are visible.
    this.pdomFocusHighlightsVisibleProperty = new BooleanProperty( true );

    // @public {BooleanProperty} - Controls whether "Interactive Highlights" are visible.
    this.interactiveHighlightsVisibleProperty = new BooleanProperty( false );

    // @public {BooleanProperty} - Controls whether "Reading Block" highlights will be visible around Nodes
    // that use ReadingBlock.
    this.readingBlockHighlightsVisibleProperty = new BooleanProperty( false );
  }

  /**
   * Set the DOM focus. A DOM limitation is that there can only be one element with focus at a time so this must
   * be a static for the FocusManager.
   * @public
   *
   * @param {Focus|null} value
   */
  static set pdomFocus( value ) {
    let previousFocus;
    if ( FocusManager.pdomFocusProperty.value ) {
      previousFocus = FocusManager.focusedNode;
    }

    FocusManager.pdomFocusProperty.value = value;

    // if set to null, make sure that the active element is no longer focused
    if ( previousFocus && !value ) {

      // blur the document.activeElement instead of going through the Node, Node.blur() won't work in cases of DAG
      document.activeElement.blur();
    }
  }

  /**
   * Get the Focus pointing to the Node whose Parallel DOM element has DOM focus.
   * @public
   *
   * @returns {Focus|null}
   */
  static get pdomFocus() { // eslint-disable-line bad-sim-text
    return FocusManager.pdomFocusProperty.value;
  }

  /**
   * Get the Node that currently has DOM focus, the leaf-most Node of the Focus Trail. Null if no
   * Node has focus.
   * @public
   *
   * @returns {Node|null}
   */
  static getPDOMFocusedNode() {
    let focusedNode = null;
    const focus = FocusManager.pdomFocusProperty.get();
    if ( focus ) {
      focusedNode = focus.trail.lastNode();
    }
    return focusedNode;
  }

  static get pdomFocusedNode() { return this.getPDOMFocusedNode(); } // eslint-disable-line bad-sim-text
}

// @public (a11y, read-only, scenery-internal settable) {Property.<Focus|null>} - Display has an axon Property to
// indicate which component is focused (or null if no scenery Node has focus). By passing the tandem and
// phetioValueType, PhET-iO is able to interoperate (save, restore, control, observe what is currently focused).
// See FocusManager.pdomFocus for setting the focus. Don't set the value of this Property directly.
FocusManager.pdomFocusProperty = new Property( null, {
  tandem: Tandem.GENERAL_MODEL.createTandem( 'domFocusProperty' ),
  phetioDocumentation: 'Stores the current focus in the simulation, null if there is nothing focused. This is not updated ' +
                       'based on mouse or touch input, only keyboard and other alternative inputs. Note that this only ' +
                       'applies to simulations that support alternative input.',
  phetioType: Property.PropertyIO( NullableIO( Focus.FocusIO ) ),
  phetioState: false,
  phetioFeatured: true,
  phetioReadOnly: true
} );

scenery.register( 'FocusManager', FocusManager );
export default FocusManager;
