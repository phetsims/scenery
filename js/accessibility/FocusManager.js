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
 *  - Pointer Focus - The Focus trail points to a Node that supports Highlighting with pointer events.
 *  - Reading Block Focus - The Focus Trail points to a Node that supports ReadingBlocks, and is active
 *                          while the ReadingBlock content is being spoken for Voicing. See ReadingBlock.js
 *
 * There may be other forms of Focus in the future.
 *
 * This class also controls setting and clearing of several (but not all) of these Properties. It does not set the
 * pdomFocusProperty because that Property is set only when the browser's focus changes. Some of the focus
 * Properties are set in feature traits, such as pointerFocusProperty which is set by MouseHighlighting because it is
 * set through listeners on each individual Node.
 *
 * This class also has a few Properties that control the behavior of the Display's HighlightOverlay.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import BooleanProperty from '../../../axon/js/BooleanProperty.js';
import DerivedProperty from '../../../axon/js/DerivedProperty.js';
import Property from '../../../axon/js/Property.js';
import Tandem from '../../../tandem/js/Tandem.js';
import NullableIO from '../../../tandem/js/types/NullableIO.js';
import scenery from '../scenery.js';
import Focus from './Focus.js';
import FocusDisplayedController from './FocusDisplayedController.js';
import ReadingBlockUtterance from './voicing/ReadingBlockUtterance.js';
import voicingManager from './voicing/voicingManager.js';

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

    // @public {DerivedProperty.<boolean>} - Indicates whether any highlights should appear from pointer
    // input (mouse/touch). If false, we will try to avoid doing expensive work in PointerHighlighting.js.
    this.pointerHighlightsVisibleProperty = new DerivedProperty(
      [ this.interactiveHighlightsVisibleProperty, voicingManager.enabledProperty ],
      ( interactiveHighlightsVisible, voicingEnabled ) => {
        return interactiveHighlightsVisible || voicingEnabled;
      } );

    //-----------------------------------------------------------------------------------------------------------------
    // The following section manages control of ReadingBlockFocusProperty. It takes a value whenever the
    // voicingManager starts speaking and the value is cleared when it stops speaking. Focus is also cleared
    // by the FocusDisplayedController.

    // @public {FocusDisplayedController} - Whenever the readingBlockFocusProperty's Focused Node is removed from
    // the scene graph or its Trail becomes invisible this removes focus.
    this.readingBlockFocusController = new FocusDisplayedController( this.readingBlockFocusProperty );

    // If the voicingManager starts speaking an Utterance for a ReadingBLock, set the readingBlockFocusProperty and
    // add listeners to clear it when the Node is removed or becomes invisible
    // @private
    this.startSpeakingListener = ( text, utterance ) => {
      this.readingBlockFocusProperty.value = utterance instanceof ReadingBlockUtterance ? utterance.readingBlockFocus : null;
    };
    voicingManager.startSpeakingEmitter.addListener( this.startSpeakingListener );

    // Whenever the voicingManager stops speaking an utterance for the ReadingBlock that has focus, clear it
    // @private
    this.endSpeakingListener = ( text, utterance ) => {
      if ( utterance instanceof ReadingBlockUtterance && this.readingBlockFocusProperty.value ) {

        // only clear the readingBlockFocusProperty if the ReadingBlockUtterance has a Focus that matches the
        // current value for readingBlockFocusProperty so that the highlight doesn't disappear every time
        // the speaker stops talking
        if ( utterance.readingBlockFocus.trail.equals( this.readingBlockFocusProperty.value.trail ) ) {
          this.readingBlockFocusProperty.value = null;
        }
      }
    };
    voicingManager.endSpeakingEmitter.addListener( this.endSpeakingListener );

    //-----------------------------------------------------------------------------------------------------------------
    // The following section manages control of pointerFocusProperty - pointerFocusProperty is set with a Focus
    // by MouseHighlighting from listeners on Nodes that use that Trait. But it uses a FocusDisplayedController
    // to remove the focus at the right time.

    this.pointerFocusDisplayedController = new FocusDisplayedController( this.pointerFocusProperty, {

      // whenever focus is removed because the last Node of the Focus Trail is no
      // longer displayed, the highlight for Pointer Focus should no longer be locked
      onRemoveFocus: () => {
        this.pointerFocusLockedProperty.value = false;
      }
    } );
  }

  /**
   * @public
   */
  dispose() {
    this.readingBlockFocusController.dispose();
    this.pointerFocusDisplayedController.dispose();
    this.pointerHighlightsVisibleProperty.dispose();
    voicingManager.startSpeakingEmitter.removeListener( this.startSpeakingListener );
    voicingManager.endSpeakingEmitter.removeListener( this.endSpeakingListener );
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
  tandem: Tandem.GENERAL_MODEL.createTandem( 'pdomFocusProperty' ),
  phetioDocumentation: 'Stores the current focus in the Parallel DOM, null if nothing has focus. This is not updated ' +
                       'based on mouse or touch input, only keyboard and other alternative inputs. Note that this only ' +
                       'applies to simulations that support alternative input.',
  phetioType: Property.PropertyIO( NullableIO( Focus.FocusIO ) ),
  phetioState: false,
  phetioFeatured: true,
  phetioReadOnly: true
} );

scenery.register( 'FocusManager', FocusManager );
export default FocusManager;
