// Copyright 2021-2025, University of Colorado Boulder

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
 *                          while the ReadingBlock content is being spoken for Voicing. See ReadingBlock.ts
 *
 * There may be other forms of Focus in the future.
 *
 * This class also controls setting and clearing of several (but not all) of these Properties. It does not set the
 * pdomFocusProperty because that Property is set only when the browser's focus changes. Some of the focus
 * Properties are set in feature traits, such as pointerFocusProperty which is set by InteractiveHighlighting because it is
 * set through listeners on each individual Node.
 *
 * This class also has a few Properties that control the behavior of the Display's HighlightOverlay.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import BooleanProperty from '../../../axon/js/BooleanProperty.js';
import DerivedProperty from '../../../axon/js/DerivedProperty.js';
import Property from '../../../axon/js/Property.js';
import TProperty from '../../../axon/js/TProperty.js';
import TReadOnlyProperty from '../../../axon/js/TReadOnlyProperty.js';
import Utterance from '../../../utterance-queue/js/Utterance.js';
import Focus from '../accessibility/Focus.js';
import FocusDisplayedController from '../accessibility/FocusDisplayedController.js';
import PDOMUtils from '../accessibility/pdom/PDOMUtils.js';
import ReadingBlockUtterance from '../accessibility/voicing/ReadingBlockUtterance.js';
import voicingManager from '../accessibility/voicing/voicingManager.js';
import type Display from '../display/Display.js';
import scenery from '../scenery.js';
import { guessVisualTrail } from './pdom/guessVisualTrail.js';
import { pdomUniqueIdToTrail } from './pdom/pdomUniqueIdToTrail.js';
import { getPDOMFocusedNode, pdomFocusProperty } from './pdomFocusProperty.js';
import { isInteractiveHighlighting } from './voicing/isInteractiveHighlighting.js';

type SpeakingListener = ( text: string, utterance: Utterance ) => void;

export default class FocusManager {

  // This Property whose Focus Trail points to the Node under the pointer to
  // support features of Voicing and Interactive Highlights. Nodes that compose InteractiveHighlighting can
  // receive this Focus and a highlight may appear around it.
  public readonly pointerFocusProperty: TProperty<Focus | null>;

  // The Property that indicates which Node that uses ReadingBlock is currently
  // active. Used by the HighlightOverlay to highlight ReadingBlock Nodes whose content is being spoken.
  public readonly readingBlockFocusProperty: TProperty<Focus | null>;

  // A Property whose value is either null or a Focus with Trail and Display equal
  // to the pointerFocusProperty. When this Property has a value, the HighlightOverlay will wait to update the
  // highlight for the pointerFocusProperty. This is useful when the pointer has begun to interact with a Node
  // that uses InteractiveHighlighting, but the mouse has moved out of it or over another during interaction. The
  // highlight should remain on the Node receiving interaction and wait to update until interaction completes.
  public readonly lockedPointerFocusProperty: TProperty<Focus | null>;

  // Controls whether or not highlights related to PDOM focus are visible.
  public readonly pdomFocusHighlightsVisibleProperty: TProperty<boolean>;

  // Controls whether "Interactive Highlights" are visible.
  public readonly interactiveHighlightsVisibleProperty: TProperty<boolean>;

  // Controls whether "Reading Block" highlights will be visible around Nodes
  // that use ReadingBlock.
  public readonly readingBlockHighlightsVisibleProperty: TProperty<boolean>;

  // Indicates whether any highlights should appear from pointer
  // input (mouse/touch). If false, we will try to avoid doing expensive work in PointerHighlighting.js.
  public readonly pointerHighlightsVisibleProperty: TReadOnlyProperty<boolean>;

  // Whenever the readingBlockFocusProperty's Focused Node is removed from
  // the scene graph or its Trail becomes invisible this removes focus.
  private readonly readingBlockFocusController: FocusDisplayedController;

  // When the lockedPointerFocusProperty's Node becomes invisible or is removed from the scene
  // graph, the locked pointer focus is cleared.
  private readonly lockedPointerFocusDisplayedController: FocusDisplayedController;

  // If the voicingManager starts speaking an Utterance for a ReadingBLock, set the readingBlockFocusProperty and
  // add listeners to clear it when the Node is removed or becomes invisible
  private readonly startSpeakingListener: SpeakingListener;

  // Whenever the voicingManager stops speaking an utterance for the ReadingBlock that has focus, clear it
  private readonly endSpeakingListener: SpeakingListener;
  private readonly pointerFocusDisplayedController: FocusDisplayedController;

  private readonly voicingFullyEnabledListener: ( enabled: boolean ) => void;

  // References to the window listeners that update when the window has focus. So they can be removed if needed.
  private static attachedWindowFocusListener: null | ( () => void ) = null;
  private static attachedWindowBlurListener: null | ( () => void ) = null;
  private static globallyAttached = false;

  public constructor() {
    this.pointerFocusProperty = new Property( null );
    this.readingBlockFocusProperty = new Property( null );
    this.lockedPointerFocusProperty = new Property( null );
    this.pdomFocusHighlightsVisibleProperty = new BooleanProperty( true );
    this.interactiveHighlightsVisibleProperty = new BooleanProperty( false );
    this.readingBlockHighlightsVisibleProperty = new BooleanProperty( false );

    // TODO: perhaps remove once reading blocks are set up to listen instead to Node.canSpeakProperty (voicingVisible), https://github.com/phetsims/scenery/issues/1343
    this.voicingFullyEnabledListener = enabled => {
      this.readingBlockHighlightsVisibleProperty.value = enabled;
    };
    voicingManager.voicingFullyEnabledProperty.link( this.voicingFullyEnabledListener );

    this.pointerHighlightsVisibleProperty = new DerivedProperty(
      [ this.interactiveHighlightsVisibleProperty, this.readingBlockHighlightsVisibleProperty ],
      ( interactiveHighlightsVisible, voicingEnabled ) => {
        return interactiveHighlightsVisible || voicingEnabled;
      } );

    //-----------------------------------------------------------------------------------------------------------------
    // The following section manages control of ReadingBlockFocusProperty. It takes a value whenever the
    // voicingManager starts speaking and the value is cleared when it stops speaking. Focus is also cleared
    // by the FocusDisplayedController.

    this.readingBlockFocusController = new FocusDisplayedController( this.readingBlockFocusProperty );

    this.startSpeakingListener = ( text, utterance ) => {
      this.readingBlockFocusProperty.value = utterance instanceof ReadingBlockUtterance ? utterance.readingBlockFocus : null;
    };

    // @ts-expect-error
    voicingManager.startSpeakingEmitter.addListener( this.startSpeakingListener );

    this.endSpeakingListener = ( text, utterance ) => {
      if ( utterance instanceof ReadingBlockUtterance && this.readingBlockFocusProperty.value ) {

        assert && assert( utterance.readingBlockFocus, 'should be non null focus' );

        // only clear the readingBlockFocusProperty if the ReadingBlockUtterance has a Focus that matches the
        // current value for readingBlockFocusProperty so that the highlight doesn't disappear every time
        // the speaker stops talking
        if ( utterance.readingBlockFocus!.trail.equals( this.readingBlockFocusProperty.value.trail ) ) {
          this.readingBlockFocusProperty.value = null;
        }
      }
    };

    // @ts-expect-error
    voicingManager.endSpeakingEmitter.addListener( this.endSpeakingListener );

    //-----------------------------------------------------------------------------------------------------------------
    // The following section manages control of pointerFocusProperty - pointerFocusProperty is set with a Focus
    // by InteractiveHighlighting from listeners on Nodes that use that Trait. But it uses a FocusDisplayedController
    // to remove the focus at the right time.
    this.pointerFocusDisplayedController = new FocusDisplayedController( this.pointerFocusProperty );
    this.lockedPointerFocusDisplayedController = new FocusDisplayedController( this.lockedPointerFocusProperty );

    [
      this.pointerFocusProperty,
      this.lockedPointerFocusProperty
    ].forEach( property => {
      property.link( this.onPointerFocusChange.bind( this ) );
    } );
  }

  public dispose(): void {
    this.pointerFocusProperty.dispose();
    this.readingBlockFocusProperty.dispose();
    this.lockedPointerFocusProperty.dispose();
    this.pdomFocusHighlightsVisibleProperty.dispose();
    this.interactiveHighlightsVisibleProperty.dispose();
    this.readingBlockHighlightsVisibleProperty.dispose();
    this.readingBlockFocusController.dispose();
    this.pointerFocusDisplayedController.dispose();
    this.pointerHighlightsVisibleProperty.dispose();
    this.lockedPointerFocusDisplayedController.dispose();

    // @ts-expect-error
    voicingManager.startSpeakingEmitter.removeListener( this.startSpeakingListener );

    // @ts-expect-error
    voicingManager.endSpeakingEmitter.removeListener( this.endSpeakingListener );

    voicingManager.voicingFullyEnabledProperty.unlink( this.voicingFullyEnabledListener );
  }

  /**
   * Update the pdomFocus from a focusin/focusout event. Scenery events are batched so that they cannot be
   * reentrant. However, that means that scenery state that needs to be updated synchronously with the
   * changing DOM cannot happen in listeners that fire from scenery input. This method
   * is meant to be called from focusin/focusout listeners on the window so that The pdomFocus matches
   * browser state.
   *
   * @param displays - List of any displays that are attached to BrowserEvents.
   * @param event - The focusin/focusout event that triggered this update.
   * @param focus - True for focusin event, false for focusout event.
   */
  public static updatePDOMFocusFromEvent( displays: Display[], event: FocusEvent, focus: boolean ): void {
    assert && assert( document.activeElement, 'Must be called from focusin, therefore active elemetn expected' );

    if ( focus ) {

      // Look for the scenery target under the PDOM
      for ( let i = 0; i < displays.length; i++ ) {
        const display = displays[ i ];

        const activeElement = document.activeElement as HTMLElement;
        if ( display.isElementUnderPDOM( activeElement, false ) ) {
          const uniqueId = activeElement.getAttribute( PDOMUtils.DATA_PDOM_UNIQUE_ID )!;
          assert && assert( uniqueId, 'Event target must have a unique ID on its data if it is in the PDOM.' );

          const trail = pdomUniqueIdToTrail( display, uniqueId )!;
          assert && assert( trail, 'We must have a trail since the target was under the PDOM.' );

          const visualTrail = guessVisualTrail( trail, display.rootNode );
          if ( visualTrail.lastNode().focusable ) {
            FocusManager.pdomFocus = new Focus( display, visualTrail );
          }
          else {

            // It is possible that `blur` or `focusout` listeners have removed the element from the traversal order
            // before we receive the `focus` event. In that case, the browser will still try to put focus on the element
            // even though the PDOM element and Node are not in the traversal order. It is more consistent to remove
            // focus in this case.
            ( event.target as HTMLElement ).blur();

            // do not allow any more focus listeners to dispatch, this target should never have been focused in the
            // first place, but the browser did it anyway
            event.stopImmediatePropagation();
          }

          // no need to keep searching
          break;
        }
      }
    }
    else {
      for ( let i = 0; i < displays.length; i++ ) {

        const display = displays[ i ];

        // will be null if it is not in the PDOM or if it is undefined
        const relatedTargetTrail = display._input!.getRelatedTargetTrail( event );

        // If there is a related target, set focus to the element that will receive focus right away. This prevents
        // the pdomFocus from being set to null. That is important for PDOMTree operations that will restore focus
        // to the next element after the PDOM is re-rendered.
        // See https://github.com/phetsims/scenery/issues/1296.
        if ( relatedTargetTrail && relatedTargetTrail.lastNode().focusable ) {
          FocusManager.pdomFocus = new Focus( display, guessVisualTrail( relatedTargetTrail, display.rootNode ) );
        }
        else {

          // Don't set this before the related target case because we want to support Node.blur listeners overwriting
          // the relatedTarget behavior.
          FocusManager.pdomFocus = null;
        }
      }
    }
  }

  // Listener to update the "active" highlight state for an interactiveHighlightingNode
  private onPointerFocusChange( pointerFocus: Focus | null, oldFocus: Focus | null ): void {
    const focusNode = pointerFocus?.trail.lastNode();
    focusNode && isInteractiveHighlighting( focusNode ) && focusNode.handleHighlightActiveChange();
    const oldFocusNode = oldFocus?.trail.lastNode();
    oldFocusNode && isInteractiveHighlighting( oldFocusNode ) && oldFocusNode.handleHighlightActiveChange();
  }

  /**
   * Set the DOM focus. A DOM limitation is that there can only be one element with focus at a time so this must
   * be a static for the FocusManager.
   */
  public static set pdomFocus( value: Focus | null ) {
    if ( pdomFocusProperty.value !== value ) {

      let previousFocus;
      if ( pdomFocusProperty.value ) {
        previousFocus = getPDOMFocusedNode();
      }

      pdomFocusProperty.value = value;

      // if set to null, make sure that the active element is no longer focused
      if ( previousFocus && !value ) {
        previousFocus.blur();
      }
    }
  }

  /**
   * Get the Focus pointing to the Node whose Parallel DOM element has DOM focus.
   */
  public static get pdomFocus(): Focus | null {
    return pdomFocusProperty.value;
  }

  /**
   // Display has an axon `Property to indicate which component is focused (or null if no
   // scenery Node has focus). By passing the tandem and phetioTye, PhET-iO is able to interoperate (save, restore,
   // control, observe what is currently focused). See FocusManager.pdomFocus for setting the focus. Don't set the value
   // of this Property directly.
   public static readonly pdomFocusProperty = new Property<Focus | null>( null, {
   tandem: Tandem.GENERAL_MODEL.createTandem( 'pdomFocusProperty' ),
   phetioDocumentation: 'Stores the current focus in the Parallel DOM, null if nothing has focus. This is not updated ' +
   'based on mouse or touch input, only keyboard and other alternative inputs. Note that this only ' +
   'applies to simulations that support alternative input.',
   phetioValueType: NullableIO( Focus.FocusIO ),
   phetioState: false,
   phetioFeatured: true,
   phetioReadOnly: true
   } );

   /**
   * A Property that lets you know when the window has focus. When the window has focus, it is in the user's foreground.
   * When in the background, the window does not receive keyboard input (important for global keyboard events).
   */
  private static _windowHasFocusProperty = new BooleanProperty( false );
  public static windowHasFocusProperty: TReadOnlyProperty<boolean> = FocusManager._windowHasFocusProperty;

  /**
   * Updates the _windowHasFocusProperty when the window receives/loses focus. When the window has focus
   * it is in the foreground of the user. When in the background, the window will not receive keyboard input.
   * https://developer.mozilla.org/en-US/docs/Web/API/Window/focus_event.
   *
   * This will be called by scenery for you when you use Display.initializeEvents().
   */
  public static attachToWindow(): void {
    assert && assert( !FocusManager.globallyAttached, 'Can only be attached statically once.' );
    FocusManager.attachedWindowFocusListener = () => {
      FocusManager._windowHasFocusProperty.value = true;
    };

    FocusManager.attachedWindowBlurListener = () => {
      FocusManager._windowHasFocusProperty.value = false;
    };

    window.addEventListener( 'focus', FocusManager.attachedWindowFocusListener );
    window.addEventListener( 'blur', FocusManager.attachedWindowBlurListener );

    // value will be updated with window, but we need a proper initial value (this function may be called while
    // the window is not in the foreground).
    FocusManager._windowHasFocusProperty.value = document.hasFocus();

    FocusManager.globallyAttached = true;
  }

  /**
   * Detach all window focus/blur listeners from FocusManager watching for when the window loses focus.
   */
  public static detachFromWindow(): void {
    window.removeEventListener( 'focus', FocusManager.attachedWindowFocusListener! );
    window.removeEventListener( 'blur', FocusManager.attachedWindowBlurListener! );

    // For cleanup, this Property becomes false again when detaching because we will no longer be watching for changes.
    FocusManager._windowHasFocusProperty.value = false;

    FocusManager.globallyAttached = false;
  }
}

scenery.register( 'FocusManager', FocusManager );