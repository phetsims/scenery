// Copyright 2021-2023, University of Colorado Boulder

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
import TProperty from '../../../axon/js/TProperty.js';
import TReadOnlyProperty from '../../../axon/js/TReadOnlyProperty.js';
import Property from '../../../axon/js/Property.js';
import Tandem from '../../../tandem/js/Tandem.js';
import NullableIO from '../../../tandem/js/types/NullableIO.js';
import Utterance from '../../../utterance-queue/js/Utterance.js';
import { Focus, FocusDisplayedController, Node, ReadingBlockUtterance, scenery, voicingManager } from '../imports.js';

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
  // that uses InteractiveHighlighting, but the mouse has moved out of it or over another during interaction. Thehighlight
  // should remain on the Node receiving interaction and wait to update until interaction completes.
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

    this.pointerFocusDisplayedController = new FocusDisplayedController( this.pointerFocusProperty, {

      // whenever focus is removed because the last Node of the Focus Trail is no
      // longer displayed, the highlight for Pointer Focus should no longer be locked
      onRemoveFocus: () => {
        this.lockedPointerFocusProperty.value = null;
      }
    } );
  }

  public dispose(): void {
    this.readingBlockFocusController.dispose();
    this.pointerFocusDisplayedController.dispose();
    this.pointerHighlightsVisibleProperty.dispose();

    // @ts-expect-error
    voicingManager.startSpeakingEmitter.removeListener( this.startSpeakingListener );

    // @ts-expect-error
    voicingManager.endSpeakingEmitter.removeListener( this.endSpeakingListener );

    voicingManager.voicingFullyEnabledProperty.unlink( this.voicingFullyEnabledListener );
  }

  /**
   * Set the DOM focus. A DOM limitation is that there can only be one element with focus at a time so this must
   * be a static for the FocusManager.
   */
  public static set pdomFocus( value: Focus | null ) {
    if ( FocusManager.pdomFocusProperty.value !== value ) {

      let previousFocus;
      if ( FocusManager.pdomFocusProperty.value ) {
        previousFocus = FocusManager.pdomFocusedNode;
      }

      FocusManager.pdomFocusProperty.value = value;

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
    return FocusManager.pdomFocusProperty.value;
  }

  /**
   * Get the Node that currently has DOM focus, the leaf-most Node of the Focus Trail. Null if no
   * Node has focus.
   */
  public static getPDOMFocusedNode(): Node | null {
    let focusedNode = null;
    const focus = FocusManager.pdomFocusProperty.get();
    if ( focus ) {
      focusedNode = focus.trail.lastNode();
    }
    return focusedNode;
  }

  public static get pdomFocusedNode(): Node | null { return this.getPDOMFocusedNode(); }

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
