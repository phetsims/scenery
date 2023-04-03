// Copyright 2017-2023, University of Colorado Boulder

/**
 * Listens to presses (down events), attaching a listener to the pointer when one occurs, so that a release (up/cancel
 * or interruption) can be recorded.
 *
 * This is the base type for both DragListener and FireListener, which contains the shared logic that would be needed
 * by both.
 *
 * PressListener is fine to use directly, particularly when drag-coordinate information is needed (e.g. DragListener),
 * or if the interaction is more complicated than a simple button fire (e.g. FireListener).
 *
 * For example usage, see scenery/examples/input.html. Additionally, a typical "simple" PressListener direct usage
 * would be something like:
 *
 *   someNode.addInputListener( new PressListener( {
 *     press: () => { ... },
 *     release: () => { ... }
 *   } ) );
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import PhetioAction from '../../../tandem/js/PhetioAction.js';
import BooleanProperty from '../../../axon/js/BooleanProperty.js';
import DerivedProperty from '../../../axon/js/DerivedProperty.js';
import EnabledComponent, { EnabledComponentOptions } from '../../../axon/js/EnabledComponent.js';
import createObservableArray, { ObservableArray } from '../../../axon/js/createObservableArray.js';
import stepTimer from '../../../axon/js/stepTimer.js';
import optionize from '../../../phet-core/js/optionize.js';
import WithoutNull from '../../../phet-core/js/types/WithoutNull.js';
import EventType from '../../../tandem/js/EventType.js';
import PhetioObject from '../../../tandem/js/PhetioObject.js';
import Tandem from '../../../tandem/js/Tandem.js';
import NullableIO from '../../../tandem/js/types/NullableIO.js';
import { Display, Mouse, Node, Pointer, scenery, SceneryEvent, TInputListener, Trail } from '../imports.js';
import TProperty from '../../../axon/js/TProperty.js';
import Bounds2 from '../../../dot/js/Bounds2.js';
import TReadOnlyProperty from '../../../axon/js/TReadOnlyProperty.js';
import IntentionalAny from '../../../phet-core/js/types/IntentionalAny.js';

// global
let globalID = 0;

// Factor out to reduce memory footprint, see https://github.com/phetsims/tandem/issues/71
const truePredicate: ( ( ...args: IntentionalAny[] ) => true ) = _.constant( true );

export type PressListenerDOMEvent = MouseEvent | TouchEvent | PointerEvent | FocusEvent | KeyboardEvent;
export type PressListenerEvent = SceneryEvent<PressListenerDOMEvent>;
export type PressListenerCallback<Listener extends PressListener> = ( event: PressListenerEvent, listener: Listener ) => void;
export type PressListenerNullableCallback<Listener extends PressListener> = ( event: PressListenerEvent | null, listener: Listener ) => void;
export type PressListenerCanStartPressCallback<Listener extends PressListener> = ( event: PressListenerEvent | null, listener: Listener ) => boolean;

type SelfOptions<Listener extends PressListener> = {
  // Called when this listener is pressed (typically from a down event, but can be triggered by other handlers)
  press?: PressListenerCallback<Listener>;

  // Called when this listener is released. Note that an SceneryEvent arg cannot be guaranteed from this listener. This
  // is, in part, to support interrupt. (pointer up/cancel or interrupt when pressed/after click from the pdom).
  // NOTE: This will also be called if the press is "released" due to being interrupted or canceled.
  release?: PressListenerNullableCallback<Listener>;

  // Called when this listener is dragged (move events on the pointer while pressed)
  drag?: PressListenerCallback<Listener>;

  // If provided, the pressedTrail (calculated from the down event) will be replaced with the (sub)trail that ends with
  // the targetNode as the leaf-most Node. This affects the parent coordinate frame computations.
  // This is ideally used when the Node which has this input listener is different from the Node being transformed,
  // as otherwise offsets and drag behavior would be incorrect by default.
  targetNode?: Node | null;

  // If true, this listener will not "press" while the associated pointer is attached, and when pressed,
  // will mark itself as attached to the pointer. If this listener should not be interrupted by others and isn't
  // a "primary" handler of the pointer's behavior, this should be set to false.
  attach?: boolean;

  // Restricts to the specific mouse button (but allows any touch). Only one mouse button is allowed at
  // a time. The button numbers are defined in https://developer.mozilla.org/en-US/docs/Web/API/MouseEvent/button,
  // where typically:
  //   0: Left mouse button
  //   1: Middle mouse button (or wheel press)
  //   2: Right mouse button
  //   3+: other specific numbered buttons that are more rare
  mouseButton?: number;

  // If the targetNode/currentTarget don't have a custom cursor, this will set the pointer cursor to
  // this value when this listener is "pressed". This means that even when the mouse moves out of the node after
  // pressing down, it will still have this cursor (overriding the cursor of whatever nodes the pointer may be
  // over).
  pressCursor?: string | null;

  // When true, any node this listener is added to will use this listener's cursor (see options.pressCursor)
  // as the cursor for that node. This only applies if the node's cursor is null, see Node.getEffectiveCursor().
  useInputListenerCursor?: boolean;

  // Checks this when trying to start a press. If this function returns false, a press will not be started
  canStartPress?: PressListenerCanStartPressCallback<Listener>;

  // (a11y) - How long something should 'look' pressed after an accessible click input event, in ms
  a11yLooksPressedInterval?: number;

  // If true, multiple drag events in a row (between steps) will be collapsed into one drag event
  // (usually for performance) by just calling the callbacks for the last drag event. Other events (press/release
  // handling) will force through the last pending drag event. Calling step() every frame will then be generally
  // necessary to have accurate-looking drags. NOTE that this may put in events out-of-order.
  // This is appropriate when the drag operation is expensive performance-wise AND ideally should only be run at
  // most once per frame (any more, and it would be a waste).
  collapseDragEvents?: boolean;

  // Though PressListener is not instrumented, declare these here to support properly passing this to children, see https://github.com/phetsims/tandem/issues/60.
  // PressListener by default doesn't allow PhET-iO to trigger press/release Action events
  phetioReadOnly?: boolean;
  phetioFeatured?: boolean;
};

export type PressListenerOptions<Listener extends PressListener = PressListener> = SelfOptions<Listener> & EnabledComponentOptions;

export type PressedPressListener = WithoutNull<PressListener, 'pointer' | 'pressedTrail'>;
const isPressedListener = ( listener: PressListener ): listener is PressedPressListener => listener.isPressed;

export default class PressListener extends EnabledComponent implements TInputListener {

  // Unique global ID for this listener
  private _id: number;

  private _mouseButton: number;
  private _a11yLooksPressedInterval: number;

  private _pressCursor: string | null;

  private _pressListener: PressListenerCallback<PressListener>;
  private _releaseListener: PressListenerNullableCallback<PressListener>;
  private _dragListener: PressListenerCallback<PressListener>;
  private _canStartPress: PressListenerCanStartPressCallback<PressListener>;

  private _targetNode: Node | null;

  private _attach: boolean;
  private _collapseDragEvents: boolean;

  // Contains all pointers that are over our button. Tracked by adding with 'enter' events and removing with 'exit'
  // events.
  public readonly overPointers: ObservableArray<Pointer>;

  // (read-only) - Tracks whether this listener is "pressed" or not.
  public readonly isPressedProperty: TProperty<boolean>;

  // (read-only) - It will be set to true when at least one pointer is over the listener.
  // This is not effected by PDOM focus.
  public readonly isOverProperty: TProperty<boolean>;

  // (read-only) - True when either isOverProperty is true, or when focused and the
  // related Display is showing its focusHighlights, see this.validateOver() for details.
  public readonly looksOverProperty: TProperty<boolean>;

  // (read-only) - It will be set to true when either:
  //   1. The listener is pressed and the pointer that is pressing is over the listener.
  //   2. There is at least one unpressed pointer that is over the listener.
  public readonly isHoveringProperty: TProperty<boolean>;

  // (read-only) - It will be set to true when either:
  //   1. The listener is pressed.
  //   2. There is at least one unpressed pointer that is over the listener.
  // This is essentially true when ( isPressed || isHovering ).
  public readonly isHighlightedProperty: TProperty<boolean>;

  // (read-only) - Whether the listener has focus (should appear to be over)
  public readonly isFocusedProperty: TProperty<boolean>;

  private readonly cursorProperty: TReadOnlyProperty<string | null>;

  // (read-only) - The current pointer, or null when not pressed. There can be short periods of
  // time when this has a value when isPressedProperty.value is false, such as during the processing of a pointer
  // release, but these periods should be very brief.
  public pointer: Pointer | null;

  // (read-only) - The Trail for the press, with no descendant nodes past the currentTarget
  // or targetNode (if provided). Will generally be null when not pressed, though there can be short periods of time
  // where this has a value when isPressedProperty.value is false, such as during the processing of a release, but
  // these periods should be very brief.
  public pressedTrail: Trail | null;

  //(read-only) - Whether the last press was interrupted. Will be valid until the next press.
  public interrupted: boolean;

  // For the collapseDragEvents feature, this will hold the last pending drag event to trigger a call to drag() with,
  // if one has been skipped.
  private _pendingCollapsedDragEvent: PressListenerEvent | null;

  // Whether our pointer listener is referenced by the pointer (need to have a flag due to handling disposal properly).
  private _listeningToPointer: boolean;

  // isHoveringProperty updates (not a DerivedProperty because we need to hook to passed-in properties)
  private _isHoveringListener: () => void;

  // isHighlightedProperty updates (not a DerivedProperty because we need to hook to passed-in properties)
  private _isHighlightedListener: () => void;

  // (read-only) - Whether a press is being processed from a pdom click input event from the PDOM.
  public readonly pdomClickingProperty: TProperty<boolean>;

  // (read-only) - This Property was added to support input from the PDOM. It tracks whether
  // or not the button should "look" down. This will be true if downProperty is true or if a pdom click is in
  // progress. For a click event from the pdom, the listeners are fired right away but the button will look down for
  // as long as a11yLooksPressedInterval. See PressListener.click() for more details.
  public readonly looksPressedProperty: TReadOnlyProperty<boolean>;

  // When pdom clicking begins, this will be added to a timeout so that the
  // pdomClickingProperty is updated after some delay. This is required since an assistive device (like a switch) may
  // send "click" events directly instead of keydown/keyup pairs. If a click initiates while already in progress,
  // this listener will be removed to start the timeout over. null until timout is added.
  private _pdomClickingTimeoutListener: ( () => void ) | null;

  // The listener that gets added to the pointer when we are pressed
  private _pointerListener: TInputListener;

  // Executed on press event
  // The main implementation of "press" handling is implemented as a callback to the PhetioAction, so things are nested
  // nicely for phet-io.
  private _pressAction: PhetioAction<[ PressListenerEvent, Node | null, ( () => void ) | null ]>;

  // Executed on release event
  // The main implementation of "release" handling is implemented as a callback to the PhetioAction, so things are nested
  // nicely for phet-io.
  private _releaseAction: PhetioAction<[ PressListenerEvent | null, ( () => void ) | null ]>;

  // To support looksOverProperty being true based on focus, we need to monitor the display from which
  // the event has come from to see if that display is showing its focusHighlights, see
  // Display.prototype.focusManager.FocusManager.pdomFocusHighlightsVisibleProperty for details.
  public display: Display | null;

  // we need the same exact function to add and remove as a listener
  private boundInvalidateOverListener: () => void;

  public constructor( providedOptions?: PressListenerOptions ) {
    const options = optionize<PressListenerOptions, SelfOptions<PressListener>, EnabledComponentOptions>()( {

      press: _.noop,
      release: _.noop,
      targetNode: null,
      drag: _.noop,
      attach: true,
      mouseButton: 0,
      pressCursor: 'pointer',
      useInputListenerCursor: false,
      canStartPress: truePredicate,
      a11yLooksPressedInterval: 100,
      collapseDragEvents: false,

      // EnabledComponent
      // By default, PressListener does not have an instrumented enabledProperty, but you can opt in with this option.
      phetioEnabledPropertyInstrumented: false,

      // phet-io (EnabledComponent)
      // For PhET-iO instrumentation. If only using the PressListener for hover behavior, there is no need to
      // instrument because events are only added to the data stream for press/release and not for hover events. Please pass
      // Tandem.OPT_OUT as the tandem option to not instrument an instance.
      tandem: Tandem.REQUIRED,

      phetioReadOnly: true,
      phetioFeatured: PhetioObject.DEFAULT_OPTIONS.phetioFeatured
    }, providedOptions );

    assert && assert( typeof options.mouseButton === 'number' && options.mouseButton >= 0 && options.mouseButton % 1 === 0,
      'mouseButton should be a non-negative integer' );
    assert && assert( options.pressCursor === null || typeof options.pressCursor === 'string',
      'pressCursor should either be a string or null' );
    assert && assert( typeof options.press === 'function',
      'The press callback should be a function' );
    assert && assert( typeof options.release === 'function',
      'The release callback should be a function' );
    assert && assert( typeof options.drag === 'function',
      'The drag callback should be a function' );
    assert && assert( options.targetNode === null || options.targetNode instanceof Node,
      'If provided, targetNode should be a Node' );
    assert && assert( typeof options.attach === 'boolean', 'attach should be a boolean' );
    assert && assert( typeof options.a11yLooksPressedInterval === 'number',
      'a11yLooksPressedInterval should be a number' );

    super( options );

    this._id = globalID++;

    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( `PressListener#${this._id} construction` );

    this._mouseButton = options.mouseButton;
    this._a11yLooksPressedInterval = options.a11yLooksPressedInterval;
    this._pressCursor = options.pressCursor;

    this._pressListener = options.press;
    this._releaseListener = options.release;
    this._dragListener = options.drag;
    this._canStartPress = options.canStartPress;

    this._targetNode = options.targetNode;

    this._attach = options.attach;
    this._collapseDragEvents = options.collapseDragEvents;

    this.overPointers = createObservableArray();

    this.isPressedProperty = new BooleanProperty( false, { reentrant: true } );
    this.isOverProperty = new BooleanProperty( false );
    this.looksOverProperty = new BooleanProperty( false );
    this.isHoveringProperty = new BooleanProperty( false );
    this.isHighlightedProperty = new BooleanProperty( false );
    this.isFocusedProperty = new BooleanProperty( false );
    this.cursorProperty = new DerivedProperty( [ this.enabledProperty ], enabled => {
      if ( options.useInputListenerCursor && enabled && this._attach ) {
        return this._pressCursor;
      }
      else {
        return null;
      }
    } );


    this.pointer = null;
    this.pressedTrail = null;
    this.interrupted = false;
    this._pendingCollapsedDragEvent = null;
    this._listeningToPointer = false;
    this._isHoveringListener = this.invalidateHovering.bind( this );
    this._isHighlightedListener = this.invalidateHighlighted.bind( this );
    this.pdomClickingProperty = new BooleanProperty( false );
    this.looksPressedProperty = DerivedProperty.or( [ this.pdomClickingProperty, this.isPressedProperty ] );
    this._pdomClickingTimeoutListener = null;
    this._pointerListener = {
      up: this.pointerUp.bind( this ),
      cancel: this.pointerCancel.bind( this ),
      move: this.pointerMove.bind( this ),
      interrupt: this.pointerInterrupt.bind( this ),
      listener: this
    };

    this._pressAction = new PhetioAction( this.onPress.bind( this ), {
      tandem: options.tandem.createTandem( 'pressAction' ),
      phetioDocumentation: 'Executes whenever a press occurs. The first argument when executing can be ' +
                           'used to convey info about the SceneryEvent.',
      phetioReadOnly: true,
      phetioFeatured: options.phetioFeatured,
      phetioEventType: EventType.USER,
      parameters: [ {
        name: 'event',
        phetioType: SceneryEvent.SceneryEventIO
      }, {
        phetioPrivate: true,
        valueType: [ Node, null ]
      }, {
        phetioPrivate: true,
        valueType: [ 'function', null ]
      }
      ]
    } );

    this._releaseAction = new PhetioAction( this.onRelease.bind( this ), {
      parameters: [ {
        name: 'event',
        phetioType: NullableIO( SceneryEvent.SceneryEventIO )
      }, {
        phetioPrivate: true,
        valueType: [ 'function', null ]
      } ],

      // phet-io
      tandem: options.tandem.createTandem( 'releaseAction' ),
      phetioDocumentation: 'Executes whenever a release occurs.',
      phetioReadOnly: true,
      phetioFeatured: options.phetioFeatured,
      phetioEventType: EventType.USER
    } );

    this.display = null;
    this.boundInvalidateOverListener = this.invalidateOver.bind( this );

    // update isOverProperty (not a DerivedProperty because we need to hook to passed-in properties)
    this.overPointers.lengthProperty.link( this.invalidateOver.bind( this ) );
    this.isFocusedProperty.link( this.invalidateOver.bind( this ) );

    // update isHoveringProperty (not a DerivedProperty because we need to hook to passed-in properties)
    this.overPointers.lengthProperty.link( this._isHoveringListener );
    this.isPressedProperty.link( this._isHoveringListener );

    // Update isHovering when any pointer's isDownProperty changes.
    // NOTE: overPointers is cleared on dispose, which should remove all of these (interior) listeners)
    this.overPointers.addItemAddedListener( pointer => pointer.isDownProperty.link( this._isHoveringListener ) );
    this.overPointers.addItemRemovedListener( pointer => pointer.isDownProperty.unlink( this._isHoveringListener ) );

    // update isHighlightedProperty (not a DerivedProperty because we need to hook to passed-in properties)
    this.isHoveringProperty.link( this._isHighlightedListener );
    this.isPressedProperty.link( this._isHighlightedListener );

    this.enabledProperty.lazyLink( this.onEnabledPropertyChange.bind( this ) );
  }

  /**
   * Whether this listener is currently activated with a press.
   */
  public get isPressed(): boolean {
    return this.isPressedProperty.value;
  }

  public get cursor(): string | null {
    return this.cursorProperty.value;
  }

  public get attach(): boolean {
    return this._attach;
  }

  public get targetNode(): Node | null {
    return this._targetNode;
  }

  /**
   * The main node that this listener is responsible for dragging.
   */
  public getCurrentTarget(): Node {
    assert && assert( this.isPressed, 'We have no currentTarget if we are not pressed' );

    return ( this as PressedPressListener ).pressedTrail.lastNode();
  }

  public get currentTarget(): Node {
    return this.getCurrentTarget();
  }

  /**
   * Returns whether a press can be started with a particular event.
   */
  public canPress( event: PressListenerEvent ): boolean {
    return !!this.enabledProperty.value &&
           !this.isPressed &&
           this._canStartPress( event, this ) &&
           // Only let presses be started with the correct mouse button.
           // @ts-expect-error Typed SceneryEvent
           ( !( event.pointer instanceof Mouse ) || event.domEvent.button === this._mouseButton ) &&
           // We can't attach to a pointer that is already attached.
           ( !this._attach || !event.pointer.isAttached() );
  }

  /**
   * Returns whether this PressListener can be clicked from keyboard input. This copies part of canPress, but
   * we didn't want to use canClick in canPress because canClick could be overridden in subtypes.
   */
  public canClick(): boolean {
    // If this listener is already involved in pressing something (or our options predicate returns false) we can't
    // press something.
    return this.enabledProperty.value && !this.isPressed && this._canStartPress( null, this );
  }

  /**
   * Moves the listener to the 'pressed' state if possible (attaches listeners and initializes press-related
   * properties).
   *
   * This can be overridden (with super-calls) when custom press behavior is needed for a type.
   *
   * This can be called by outside clients in order to try to begin a process (generally on an already-pressed
   * pointer), and is useful if a 'drag' needs to change between listeners. Use canPress( event ) to determine if
   * a press can be started (if needed beforehand).
   *
   * @param event
   * @param [targetNode] - If provided, will take the place of the targetNode for this call. Useful for
   *                              forwarded presses.
   * @param [callback] - to be run at the end of the function, but only on success
   * @returns success - Returns whether the press was actually started
   */
  public press( event: PressListenerEvent, targetNode?: Node, callback?: () => void ): boolean {
    assert && assert( event, 'An event is required' );

    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( `PressListener#${this._id} press` );

    if ( !this.canPress( event ) ) {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( `PressListener#${this._id} could not press` );
      return false;
    }

    // Flush out a pending drag, so it happens before we press
    this.flushCollapsedDrag();

    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( `PressListener#${this._id} successful press` );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();
    this._pressAction.execute( event, targetNode || null, callback || null ); // cannot pass undefined into execute call

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();

    return true;
  }

  /**
   * Releases a pressed listener.
   *
   * This can be overridden (with super-calls) when custom release behavior is needed for a type.
   *
   * This can be called from the outside to release the press without the pointer having actually fired any 'up'
   * events. If the cancel/interrupt behavior is more preferable, call interrupt() on this listener instead.
   *
   * @param [event] - scenery event if there was one. We can't guarantee an event, in part to support interrupting.
   * @param [callback] - called at the end of the release
   */
  public release( event?: PressListenerEvent, callback?: () => void ): void {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( `PressListener#${this._id} release` );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    // Flush out a pending drag, so it happens before we release
    this.flushCollapsedDrag();

    this._releaseAction.execute( event || null, callback || null ); // cannot pass undefined to execute call

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  /**
   * Called when move events are fired on the attached pointer listener.
   *
   * This can be overridden (with super-calls) when custom drag behavior is needed for a type.
   *
   * (scenery-internal, effectively protected)
   */
  public drag( event: PressListenerEvent ): void {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( `PressListener#${this._id} drag` );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    assert && assert( this.isPressed, 'Can only drag while pressed' );

    this._dragListener( event, this );

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  /**
   * Interrupts the listener, releasing it (canceling behavior).
   *
   * This effectively releases/ends the press, and sets the `interrupted` flag to true while firing these events
   * so that code can determine whether a release/end happened naturally, or was canceled in some way.
   *
   * This can be called manually, but can also be called through node.interruptSubtreeInput().
   */
  public interrupt(): void {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( `PressListener#${this._id} interrupt` );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    // handle pdom interrupt
    if ( this.pdomClickingProperty.value ) {
      this.interrupted = true;

      // it is possible we are interrupting a click with a pointer press, in which case
      // we are listening to the Pointer listener - do a full release in this case
      if ( this._listeningToPointer ) {
        this.release();
      }
      else {

        // release on interrupt (without going through onRelease, which handles mouse/touch specific things)
        this.isPressedProperty.value = false;
        this._releaseListener( null, this );
      }

      // clear the clicking timer, specific to pdom input
      // @ts-expect-error TODO: This looks buggy, will need to ignore for now
      if ( stepTimer.hasListener( this._pdomClickingTimeoutListener ) ) {
        // @ts-expect-error TODO: This looks buggy, will need to ignore for now
        stepTimer.clearTimeout( this._pdomClickingTimeoutListener );

        // interrupt may be called after the PressListener has been disposed (for instance, internally by scenery
        // if the Node receives a blur event after the PressListener is disposed)
        if ( !this.pdomClickingProperty.isDisposed ) {
          this.pdomClickingProperty.value = false;
        }
      }
    }
    else if ( this.isPressed ) {

      // handle pointer interrupt
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( `PressListener#${this._id} interrupting` );
      this.interrupted = true;

      this.release();
    }

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  /**
   * This should be called when the listened "Node" is effectively removed from the scene graph AND
   * expected to be placed back in such that it could potentially get multiple "enter" events, see
   * https://github.com/phetsims/scenery/issues/1021
   *
   * This will clear the list of pointers considered "over" the Node, so that when it is placed back in, the state
   * will be correct, and another "enter" event will not be missing an "exit".
   */
  public clearOverPointers(): void {
    this.overPointers.clear(); // We have listeners that will trigger the proper refreshes
  }

  /**
   * If collapseDragEvents is set to true, this step() should be called every frame so that the collapsed drag
   * can be fired.
   */
  public step(): void {
    this.flushCollapsedDrag();
  }

  /**
   * Set the callback that will create a Bounds2 in the global coordinate frame for the AnimatedPanZoomListener to
   * keep in view during a drag operation. During drag input the AnimatedPanZoomListener will pan the screen to
   * try and keep the returned Bounds2 visible. By default, the AnimatedPanZoomListener will try to keep the target of
   * the drag in view but that may not always work if the target is not associated with the translated Node, the target
   * is not defined, or the target has bounds that do not accurately surround the graphic you want to keep in view.
   */
  public setCreatePanTargetBounds( createDragPanTargetBounds: ( () => Bounds2 ) | null ): void {

    // Forwarded to the pointerListener so that the AnimatedPanZoomListener can get this callback from the attached
    // listener
    this._pointerListener.createPanTargetBounds = createDragPanTargetBounds;
  }

  public set createPanTargetBounds( createDragPanTargetBounds: ( () => Bounds2 ) | null ) { this.setCreatePanTargetBounds( createDragPanTargetBounds ); }

  /**
   * A convenient way to create and set the callback that will return a Bounds2 in the global coordinate frame for the
   * AnimatedPanZoomListener to keep in view during a drag operation. The AnimatedPanZoomListener will try to keep the
   * bounds of the last Node of the provided trail visible by panning the screen during a drag operation. See
   * setCreatePanTargetBounds() for more documentation.
   */
  public setCreatePanTargetBoundsFromTrail( trail: Trail ): void {
    assert && assert( trail.length > 0, 'trail has no Nodes to provide localBounds' );
    this.setCreatePanTargetBounds( () => trail.localToGlobalBounds( trail.lastNode().localBounds ) );
  }

  public set createPanTargetBoundsFromTrail( trail: Trail ) { this.setCreatePanTargetBoundsFromTrail( trail ); }

  /**
   * If there is a pending collapsed drag waiting, we'll fire that drag (usually before other events or during a step)
   */
  private flushCollapsedDrag(): void {
    if ( this._pendingCollapsedDragEvent ) {
      this.drag( this._pendingCollapsedDragEvent );
    }
    this._pendingCollapsedDragEvent = null;
  }

  /**
   * Recomputes the value for isOverProperty. Separate to reduce anonymous function closures.
   */
  private invalidateOver(): void {
    let pointerAttachedToOther = false;

    if ( this._listeningToPointer ) {

      // this pointer listener is attached to the pointer
      pointerAttachedToOther = false;
    }
    else {

      // a listener other than this one is attached to the pointer so it should not be considered over
      for ( let i = 0; i < this.overPointers.length; i++ ) {
        if ( this.overPointers.get( i )!.isAttached() ) {
          pointerAttachedToOther = true;
          break;
        }
      }
    }

    // isOverProperty is only for the `over` event, looksOverProperty includes focused pressListeners (only when the
    // display is showing focus highlights)
    this.isOverProperty.value = ( this.overPointers.length > 0 && !pointerAttachedToOther );
    this.looksOverProperty.value = this.isOverProperty.value ||
                                   ( this.isFocusedProperty.value && !!this.display && this.display.focusManager.pdomFocusHighlightsVisibleProperty.value );
  }

  /**
   * Recomputes the value for isHoveringProperty. Separate to reduce anonymous function closures.
   */
  private invalidateHovering(): void {
    for ( let i = 0; i < this.overPointers.length; i++ ) {
      const pointer = this.overPointers[ i ];
      if ( !pointer.isDown || pointer === this.pointer ) {
        this.isHoveringProperty.value = true;
        return;
      }
    }
    this.isHoveringProperty.value = false;
  }

  /**
   * Recomputes the value for isHighlightedProperty. Separate to reduce anonymous function closures.
   */
  private invalidateHighlighted(): void {
    this.isHighlightedProperty.value = this.isHoveringProperty.value || this.isPressedProperty.value;
  }

  /**
   * Fired when the enabledProperty changes
   */
  protected onEnabledPropertyChange( enabled: boolean ): void {
    !enabled && this.interrupt();
  }

  /**
   * Internal code executed as the first step of a press.
   *
   * @param event
   * @param [targetNode] - If provided, will take the place of the targetNode for this call. Useful for
   *                              forwarded presses.
   * @param [callback] - to be run at the end of the function, but only on success
   */
  private onPress( event: PressListenerEvent, targetNode: Node | null, callback: ( () => void ) | null ): void {
    assert && assert( !this.isDisposed, 'Should not press on a disposed listener' );

    const givenTargetNode = targetNode || this._targetNode;

    // Set this properties before the property change, so they are visible to listeners.
    this.pointer = event.pointer;
    this.pressedTrail = givenTargetNode ? givenTargetNode.getUniqueTrail() : event.trail.subtrailTo( event.currentTarget!, false );

    this.interrupted = false; // clears the flag (don't set to false before here)

    this.pointer.addInputListener( this._pointerListener, this._attach );
    this._listeningToPointer = true;

    this.pointer.cursor = this.pressedTrail.lastNode().getEffectiveCursor() || this._pressCursor;

    this.isPressedProperty.value = true;

    // Notify after everything else is set up
    this._pressListener( event, this );

    callback && callback();
  }

  /**
   * Internal code executed as the first step of a release.
   *
   * @param event - scenery event if there was one
   * @param [callback] - called at the end of the release
   */
  private onRelease( event: PressListenerEvent | null, callback: ( () => void ) | null ): void {
    assert && assert( this.isPressed, 'This listener is not pressed' );
    const pressedListener = this as PressedPressListener;

    pressedListener.pointer.removeInputListener( this._pointerListener );
    this._listeningToPointer = false;

    // Set the pressed state false *before* invoking the callback, otherwise an infinite loop can result in some
    // circumstances.
    this.isPressedProperty.value = false;

    // Notify after the rest of release is called in order to prevent it from triggering interrupt().
    this._releaseListener( event, this );

    callback && callback();

    // These properties are cleared now, at the end of the onRelease, in case they were needed by the callback or in
    // listeners on the pressed Property.
    pressedListener.pointer.cursor = null;
    this.pointer = null;
    this.pressedTrail = null;
  }

  /**
   * Called with 'down' events (part of the listener API). (scenery-internal)
   *
   * NOTE: Do not call directly. See the press method instead.
   */
  public down( event: PressListenerEvent ): void {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( `PressListener#${this._id} down` );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    this.press( event );

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  /**
   * Called with 'up' events (part of the listener API). (scenery-internal)
   *
   * NOTE: Do not call directly.
   */
  public up( event: PressListenerEvent ): void {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( `PressListener#${this._id} up` );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    // Recalculate over/hovering Properties.
    this.invalidateOver();
    this.invalidateHovering();

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  /**
   * Called with 'enter' events (part of the listener API). (scenery-internal)
   *
   * NOTE: Do not call directly.
   */
  public enter( event: PressListenerEvent ): void {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( `PressListener#${this._id} enter` );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    this.overPointers.push( event.pointer );

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  /**
   * Called with `move` events (part of the listener API). It is necessary to check for `over` state changes on move
   * in case a pointer listener gets interrupted and resumes movement over a target. (scenery-internal)
   */
  public move( event: PressListenerEvent ): void {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( `PressListener#${this._id} move` );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    this.invalidateOver();

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  /**
   * Called with 'exit' events (part of the listener API). (scenery-internal)
   *
   * NOTE: Do not call directly.
   */
  public exit( event: PressListenerEvent ): void {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( `PressListener#${this._id} exit` );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    // NOTE: We don't require the pointer to be included here, since we may have added the listener after the 'enter'
    // was fired. See https://github.com/phetsims/area-model-common/issues/159 for more details.
    if ( this.overPointers.includes( event.pointer ) ) {
      this.overPointers.remove( event.pointer );
    }

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  /**
   * Called with 'up' events from the pointer (part of the listener API) (scenery-internal)
   *
   * NOTE: Do not call directly.
   */
  public pointerUp( event: PressListenerEvent ): void {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( `PressListener#${this._id} pointer up` );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    // Since our callback can get queued up and THEN interrupted before this happens, we'll check to make sure we are
    // still pressed by the time we get here. If not pressed, then there is nothing to do.
    // See https://github.com/phetsims/capacitor-lab-basics/issues/251
    if ( this.isPressed ) {
      assert && assert( event.pointer === this.pointer );

      this.release( event );
    }

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  /**
   * Called with 'cancel' events from the pointer (part of the listener API) (scenery-internal)
   *
   * NOTE: Do not call directly.
   */
  public pointerCancel( event: PressListenerEvent ): void {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( `PressListener#${this._id} pointer cancel` );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    // Since our callback can get queued up and THEN interrupted before this happens, we'll check to make sure we are
    // still pressed by the time we get here. If not pressed, then there is nothing to do.
    // See https://github.com/phetsims/capacitor-lab-basics/issues/251
    if ( this.isPressed ) {
      assert && assert( event.pointer === this.pointer );

      this.interrupt(); // will mark as interrupted and release()
    }

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  /**
   * Called with 'move' events from the pointer (part of the listener API) (scenery-internal)
   *
   * NOTE: Do not call directly.
   */
  public pointerMove( event: PressListenerEvent ): void {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( `PressListener#${this._id} pointer move` );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    // Since our callback can get queued up and THEN interrupted before this happens, we'll check to make sure we are
    // still pressed by the time we get here. If not pressed, then there is nothing to do.
    // See https://github.com/phetsims/capacitor-lab-basics/issues/251
    if ( this.isPressed ) {
      assert && assert( event.pointer === this.pointer );

      if ( this._collapseDragEvents ) {
        this._pendingCollapsedDragEvent = event;
      }
      else {
        this.drag( event );
      }
    }

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  /**
   * Called when the pointer needs to interrupt its current listener (usually so another can be added). (scenery-internal)
   *
   * NOTE: Do not call directly.
   */
  public pointerInterrupt(): void {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( `PressListener#${this._id} pointer interrupt` );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    this.interrupt();

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  /**
   * Click listener, called when this is treated as an accessible input listener.
   * In general not needed to be public, but just used in edge cases to get proper click logic for pdom.
   *
   * Handle the click event from DOM for PDOM. Clicks by calling press and release immediately.
   * When assistive technology is used, the browser may not receive 'keydown' or 'keyup' events on input elements, but
   * only a single 'click' event. We need to toggle the pressed state from the single 'click' event.
   *
   * This will fire listeners immediately, but adds a delay for the pdomClickingProperty so that you can make a
   * button look pressed from a single DOM click event. For example usage, see sun/ButtonModel.looksPressedProperty.
   *
   * @param event
   * @param [callback] optionally called immediately after press, but only on successful click
   * @returns success - Returns whether the press was actually started
   */
  public click( event: SceneryEvent<MouseEvent> | null, callback?: () => void ): boolean {
    if ( this.canClick() ) {
      this.interrupted = false; // clears the flag (don't set to false before here)

      this.pdomClickingProperty.value = true;

      // ensure that button is 'focused' so listener can be called while button is down
      this.isFocusedProperty.value = true;
      this.isPressedProperty.value = true;

      // fire the optional callback
      // @ts-expect-error
      this._pressListener( event, this );

      callback && callback();

      // no longer down, don't reset 'over' so button can be styled as long as it has focus
      this.isPressedProperty.value = false;

      // fire the callback from options
      this._releaseListener( event, this );

      // if we are already clicking, remove the previous timeout - this assumes that clearTimeout is a noop if the
      // listener is no longer attached
      // @ts-expect-error TODO: This looks buggy, will need to ignore for now
      stepTimer.clearTimeout( this._pdomClickingTimeoutListener );

      // Now add the timeout back to start over, saving so that it can be removed later. Even when this listener was
      // interrupted from above logic, we still delay setting this to false to support visual "pressing" redraw.
      // @ts-expect-error TODO: This looks buggy, will need to ignore for now
      this._pdomClickingTimeoutListener = stepTimer.setTimeout( () => {

        // the listener may have been disposed before the end of a11yLooksPressedInterval, like if it fires and
        // disposes itself immediately
        if ( !this.pdomClickingProperty.isDisposed ) {
          this.pdomClickingProperty.value = false;
        }
      }, this._a11yLooksPressedInterval );
    }

    return true;
  }

  /**
   * Focus listener, called when this is treated as an accessible input listener and its target is focused. (scenery-internal)
   * @pdom
   */
  public focus( event: SceneryEvent<FocusEvent> ): void {

    // Get the Display related to this accessible event.
    const accessibleDisplays = event.trail.rootNode().getRootedDisplays().filter( display => display.isAccessible() );
    assert && assert( accessibleDisplays.length === 1,
      'cannot focus node with zero or multiple accessible displays attached' );
    //
    this.display = accessibleDisplays[ 0 ];
    if ( !this.display.focusManager.pdomFocusHighlightsVisibleProperty.hasListener( this.boundInvalidateOverListener ) ) {
      this.display.focusManager.pdomFocusHighlightsVisibleProperty.link( this.boundInvalidateOverListener );
    }

    // On focus, button should look 'over'.
    this.isFocusedProperty.value = true;
  }

  /**
   * Blur listener, called when this is treated as an accessible input listener.
   * @pdom
   */
  public blur(): void {
    if ( this.display ) {
      if ( this.display.focusManager.pdomFocusHighlightsVisibleProperty.hasListener( this.boundInvalidateOverListener ) ) {
        this.display.focusManager.pdomFocusHighlightsVisibleProperty.unlink( this.boundInvalidateOverListener );
      }
      this.display = null;
    }

    // On blur, the button should no longer look 'over'.
    this.isFocusedProperty.value = false;
  }

  /**
   * Disposes the listener, releasing references. It should not be used after this.
   */
  public override dispose(): void {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( `PressListener#${this._id} dispose` );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    // We need to release references to any pointers that are over us.
    this.overPointers.clear();

    if ( this._listeningToPointer && isPressedListener( this ) ) {
      this.pointer.removeInputListener( this._pointerListener );
    }

    // These Properties could have already been disposed, for example in the sun button hierarchy, see https://github.com/phetsims/sun/issues/372
    if ( !this.isPressedProperty.isDisposed ) {
      this.isPressedProperty.unlink( this._isHighlightedListener );
      this.isPressedProperty.unlink( this._isHoveringListener );
    }
    !this.isHoveringProperty.isDisposed && this.isHoveringProperty.unlink( this._isHighlightedListener );

    this._pressAction.dispose();
    this._releaseAction.dispose();

    this.looksPressedProperty.dispose();
    this.pdomClickingProperty.dispose();
    this.cursorProperty.dispose();
    this.isFocusedProperty.dispose();
    this.isHighlightedProperty.dispose();
    this.isHoveringProperty.dispose();
    this.looksOverProperty.dispose();
    this.isOverProperty.dispose();
    this.isPressedProperty.dispose();
    this.overPointers.dispose();

    // Remove references to the stored display, if we have any.
    if ( this.display ) {
      this.display.focusManager.pdomFocusHighlightsVisibleProperty.unlink( this.boundInvalidateOverListener );
      this.display = null;
    }

    super.dispose();

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  public static phetioAPI = {
    pressAction: { phetioType: PhetioAction.PhetioActionIO( [ SceneryEvent.SceneryEventIO ] ) },
    releaseAction: { phetioType: PhetioAction.PhetioActionIO( [ NullableIO( SceneryEvent.SceneryEventIO ) ] ) }
  };
}

scenery.register( 'PressListener', PressListener );
