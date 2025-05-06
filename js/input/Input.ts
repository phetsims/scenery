// Copyright 2013-2025, University of Colorado Boulder

/**
 * Main handler for user-input events in Scenery.
 *
 * *** Adding input handling to a display
 *
 * Displays do not have event listeners attached by default. To initialize the event system (that will set up
 * listeners), use one of Display's initialize*Events functions.
 *
 * *** Pointers
 *
 * A 'pointer' is an abstract way of describing a mouse, a single touch point, or a pen/stylus, similar to in the
 * Pointer Events specification (https://dvcs.w3.org/hg/pointerevents/raw-file/tip/pointerEvents.html). Touch and pen
 * pointers are transient, created when the relevant DOM down event occurs and released when corresponding the DOM up
 * or cancel event occurs. However, the mouse pointer is persistent.
 *
 * Input event listeners can be added to {Node}s directly, or to a pointer. When a DOM event is received, it is first
 * broken up into multiple events (if necessary, e.g. multiple touch points), then the dispatch is handled for each
 * individual Scenery event. Events are first fired for any listeners attached to the pointer that caused the event,
 * then fire on the node directly under the pointer, and if applicable, bubble up the graph to the Scene from which the
 * event was triggered. Events are not fired directly on nodes that are not under the pointer at the time of the event.
 * To handle many common patterns (like button presses, where mouse-ups could happen when not over the button), it is
 * necessary to add those move/up listeners to the pointer itself.
 *
 * *** Listeners and Events
 *
 * Event listeners are added with node.addInputListener( listener ), pointer.addInputListener( listener ) and
 * display.addInputListener( listener ).
 * This listener can be an arbitrary object, and the listener will be triggered by calling listener[eventType]( event ),
 * where eventType is one of the event types as described below, and event is a Scenery event with the
 * following properties:
 * - trail {Trail} - Points to the node under the pointer
 * - pointer {Pointer} - The pointer that triggered the event. Additional information about the mouse/touch/pen can be
 *                       obtained from the pointer, for example event.pointer.point.
 * - type {string} - The base type of the event (e.g. for touch down events, it will always just be "down").
 * - domEvent {UIEvent} - The underlying DOM event that triggered this Scenery event. The DOM event may correspond to
 *                        multiple Scenery events, particularly for touch events. This could be a TouchEvent,
 *                        PointerEvent, MouseEvent, MSPointerEvent, etc.
 * - target {Node} - The leaf-most Node in the trail.
 * - currentTarget {Node} - The Node to which the listener being fired is attached, or null if the listener is being
 *                          fired directly from a pointer.
 *
 * Additionally, listeners may support an interrupt() method that detaches it from pointers, or may support being
 * "attached" to a pointer (indicating a primary role in controlling the pointer's behavior). See Pointer for more
 * information about these interactions.
 *
 * *** Event Types
 *
 * Scenery will fire the following base event types:
 *
 * - down: Triggered when a pointer is pressed down. Touch / pen pointers are created for each down event, and are
 *         active until an up/cancel event is sent.
 * - up: Triggered when a pointer is released normally. Touch / pen pointers will not have any more events associated
 *       with them after an up event.
 * - cancel: Triggered when a pointer is canceled abnormally. Touch / pen pointers will not have any more events
 *           associated with them after an up event.
 * - move: Triggered when a pointer moves.
 * - wheel: Triggered when the (mouse) wheel is scrolled. The associated pointer will have wheelDelta information.
 * - enter: Triggered when a pointer moves over a Node or one of its children. Does not bubble up. Mirrors behavior from
 *          the DOM mouseenter (http://www.w3.org/TR/DOM-Level-3-Events/#event-type-mouseenter)
 * - exit:  Triggered when a pointer moves out from over a Node or one of its children. Does not bubble up. Mirrors
 *          behavior from the DOM mouseleave (http://www.w3.org/TR/DOM-Level-3-Events/#event-type-mouseleave).
 * - over: Triggered when a pointer moves over a Node (not including its children). Mirrors behavior from the DOM
 *         mouseover (http://www.w3.org/TR/DOM-Level-3-Events/#event-type-mouseover).
 * - out: Triggered when a pointer moves out from over a Node (not including its children). Mirrors behavior from the
 *        DOM mouseout (http://www.w3.org/TR/DOM-Level-3-Events/#event-type-mouseout).
 *
 * Before firing the base event type (for example, 'move'), Scenery will also fire an event specific to the type of
 * pointer. For mice, it will fire 'mousemove', for touch events it will fire 'touchmove', and for pen events it will
 * fire 'penmove'. Similarly, for any type of event, it will first fire pointerType+eventType, and then eventType.
 *
 * **** PDOM Specific Event Types
 *
 * Some event types can only be triggered from the PDOM. If a SCENERY/Node has accessible content (see
 * ParallelDOM.js for more info), then listeners can be added for events fired from the PDOM. The accessibility events
 * triggered from a Node are dependent on the `tagName` (ergo the HTMLElement primary sibling) specified by the Node.
 *
 * Some terminology for understanding:
 * - PDOM:  parallel DOM, see ParallelDOM.js
 * - Primary Sibling:  The Node's HTMLElement in the PDOM that is interacted with for accessible interactions and to
 *                     display accessible content. The primary sibling has the tag name specified by the `tagName`
 *                     option, see `ParallelDOM.setTagName`. Primary sibling is further defined in PDOMPeer.js
 * - Assistive Technology:  aka AT, devices meant to improve the capabilities of an individual with a disability.
 *
 * The following are the supported accessible events:
 *
 * - focus: Triggered when navigation focus is set to this Node's primary sibling. This can be triggered with some
 *          AT too, like screen readers' virtual cursor, but that is not dependable as it can be toggled with a screen
 *          reader option. Furthermore, this event is not triggered on mobile devices. Does not bubble.
 * - focusin: Same as 'focus' event, but bubbles.
 * - blur:  Triggered when navigation focus leaves this Node's primary sibling. This can be triggered with some
 *          AT too, like screen readers' virtual cursor, but that is not dependable as it can be toggled with a screen
 *          reader option. Furthermore, this event is not triggered on mobile devices.
 * - focusout: Same as 'blur' event, but bubbles.
 * - click:  Triggered when this Node's primary sibling is clicked. Note, though this event seems similar to some base
 *           event types (the event implements `MouseEvent`), it only applies when triggered from the PDOM.
 *           See https://www.w3.org/TR/DOM-Level-3-Events/#click
 * - input:  Triggered when the value of an <input>, <select>, or <textarea> element has been changed.
 *           See https://www.w3.org/TR/DOM-Level-3-Events/#input
 * - change:  Triggered for <input>, <select>, and <textarea> elements when an alteration to the element's value is
 *            committed by the user. Unlike the input event, the change event is not necessarily fired for each
 *            alteration to an element's value. See
 *            https://developer.mozilla.org/en-US/docs/Web/API/HTMLElement/change_event and
 *            https://html.spec.whatwg.org/multipage/indices.html#event-change
 * - keydown: Triggered for all keys pressed. When a screen reader is active, this event will be omitted
 *            role="button" is activated.
 *            See https://www.w3.org/TR/DOM-Level-3-Events/#keydown
 * - keyup :  Triggered for all keys when released. When a screen reader is active, this event will be omitted
 *            role="button" is activated.
 *            See https://www.w3.org/TR/DOM-Level-3-Events/#keyup
 *
 * *** Event Dispatch
 *
 * Events have two methods that will cause early termination: event.abort() will cause no more listeners to be notified
 * for this event, and event.handle() will allow the current level of listeners to be notified (all pointer listeners,
 * or all listeners attached to the current node), but no more listeners after that level will fire. handle and abort
 * are like stopPropagation, stopImmediatePropagation for DOM events, except they do not trigger those DOM methods on
 * the underlying DOM event.
 *
 * Up/down/cancel events all happen separately, but for move events, a specific sequence of events occurs if the pointer
 * changes the node it is over:
 *
 * 1. The move event is fired (and bubbles).
 * 2. An out event is fired for the old topmost Node (and bubbles).
 * 3. exit events are fired for all Nodes in the Trail hierarchy that are now not under the pointer, from the root-most
 *    to the leaf-most. Does not bubble.
 * 4. enter events are fired for all Nodes in the Trail hierarchy that were not under the pointer (but now are), from
 *    the leaf-most to the root-most. Does not bubble.
 * 5. An over event is fired for the new topmost Node (and bubbles).
 *
 * event.abort() and event.handle() will currently not affect other stages in the 'move' sequence (e.g. event.abort() in
 * the 'move' event will not affect the following 'out' event).
 *
 * For each event type:
 *
 * 1. Listeners on the pointer will be triggered first (in the order they were added)
 * 2. Listeners on the target (top-most) Node will be triggered (in the order they were added to that Node)
 * 3. Then if the event bubbles, each Node in the Trail will be triggered, starting from the Node under the top-most
 *    (that just had listeners triggered) and all the way down to the Scene. Listeners are triggered in the order they
 *    were added for each Node.
 * 4. Listeners on the display will be triggered (in the order they were added)
 *
 * For each listener being notified, it will fire the more specific pointerType+eventType first (e.g. 'mousemove'),
 * then eventType next (e.g. 'move').
 *
 * Currently, preventDefault() is called on the associated DOM event if the top-most node has the 'interactive' property
 * set to a truthy value.
 *
 * *** Relevant Specifications
 *
 * DOM Level 3 events spec: http://www.w3.org/TR/DOM-Level-3-Events/
 * Touch events spec: http://www.w3.org/TR/touch-events/
 * Pointer events spec draft: https://dvcs.w3.org/hg/pointerevents/raw-file/tip/pointerEvents.html
 *                            http://msdn.microsoft.com/en-us/library/ie/hh673557(v=vs.85).aspx
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @author Sam Reid (PhET Interactive Simulations)
 */

import stepTimer from '../../../axon/js/stepTimer.js';
import TEmitter from '../../../axon/js/TEmitter.js';
import TinyEmitter from '../../../axon/js/TinyEmitter.js';
import Vector2 from '../../../dot/js/Vector2.js';
import cleanArray from '../../../phet-core/js/cleanArray.js';
import optionize, { EmptySelfOptions } from '../../../phet-core/js/optionize.js';
import platform from '../../../phet-core/js/platform.js';
import PickOptional from '../../../phet-core/js/types/PickOptional.js';
import EventType from '../../../tandem/js/EventType.js';
import PhetioAction from '../../../tandem/js/PhetioAction.js';
import PhetioObject, { PhetioObjectOptions } from '../../../tandem/js/PhetioObject.js';
import ArrayIO from '../../../tandem/js/types/ArrayIO.js';
import IOType from '../../../tandem/js/types/IOType.js';
import NullableIO from '../../../tandem/js/types/NullableIO.js';
import NumberIO from '../../../tandem/js/types/NumberIO.js';
import { pdomUniqueIdToTrail } from '../accessibility/pdom/pdomUniqueIdToTrail.js';
import PDOMUtils from '../accessibility/pdom/PDOMUtils.js';
import type Display from '../display/Display.js';
import DisplayGlobals from '../display/DisplayGlobals.js';
import BatchedDOMEvent, { BatchedDOMEventCallback, BatchedDOMEventType } from '../input/BatchedDOMEvent.js';
import BrowserEvents from '../input/BrowserEvents.js';
import EventContext, { EventContextIO } from '../input/EventContext.js';
import Mouse from '../input/Mouse.js';
import PDOMPointer from '../input/PDOMPointer.js';
import Pen from '../input/Pen.js';
import Pointer, { PointerState } from '../input/Pointer.js';
import SceneryEvent from '../input/SceneryEvent.js';
import type { default as TInputListener, SceneryListenerFunction, SupportedEventTypes } from '../input/TInputListener.js';
import Touch from '../input/Touch.js';
import WindowTouch from '../input/WindowTouch.js';
import Node from '../nodes/Node.js';
import scenery from '../scenery.js';
import Trail from '../util/Trail.js';
import { TARGET_SUBSTITUTE_KEY } from './eventSerialization.js';

const ArrayIOPointerIO = ArrayIO( Pointer.PointerIO );

// A list of events that will still fire, even when the Node does not have input enabled or is not pickable.
const PDOM_ALWAYS_ACTIVE_EVENTS = [ 'focus', 'blur', 'focusin', 'focusout' ];

type TargetSubstitudeAugmentedEvent = Event & {
  [ TARGET_SUBSTITUTE_KEY ]?: Element;
};

type InputState = {
  pointers: PointerState[];
};


// A bit more than the maximum amount of time that iOS 14 VoiceOver was observed to delay between
// sending a mouseup event and a click event.
const PDOM_CLICK_DELAY = 80;

type SelfOptions = EmptySelfOptions;

export type InputOptions = SelfOptions & PickOptional<PhetioObjectOptions, 'tandem'>;

export default class Input extends PhetioObject {

  public readonly display: Display;
  public readonly rootNode: Node;

  public readonly attachToWindow: boolean;
  public readonly batchDOMEvents: boolean;
  public readonly assumeFullWindow: boolean;
  public readonly passiveEvents: boolean | null;
  public readonly preventMultitouch: boolean;
  public readonly interruptMultitouch: boolean;

  // Pointer for accessibility, only created lazily on first pdom event.
  public pdomPointer: PDOMPointer | null;

  // Pointer for mouse, only created lazily on first mouse event, so no mouse is allocated on tablets.
  public mouse: Mouse | null;

  // All active pointers.
  public pointers: Pointer[];

  public pointerAddedEmitter: TEmitter<[ Pointer ]>;

  // Whether we are currently firing events. We need to track this to handle re-entrant cases
  // like https://github.com/phetsims/balloons-and-static-electricity/issues/406.
  public currentlyFiringEvents: boolean;

  public currentSceneryEvent: SceneryEvent | null = null;

  private batchedEvents: BatchedDOMEvent[];

  // In miliseconds, the DOMEvent timeStamp when we receive a logical up event.
  // We can compare this to the timeStamp on a click vent to filter out the click events
  // when some screen readers send both down/up events AND click events to the target
  // element, see https://github.com/phetsims/scenery/issues/1094
  private upTimeStamp: number;

  // Emits pointer validation to the input stream for playback
  // This is a high frequency event that is necessary for reproducible playbacks
  private readonly validatePointersAction: PhetioAction;

  private readonly mouseUpAction: PhetioAction<[ Vector2, EventContext<MouseEvent> ]>;
  private readonly mouseDownAction: PhetioAction<[ number, Vector2, EventContext<MouseEvent> ]>;
  private readonly mouseMoveAction: PhetioAction<[ Vector2, EventContext<MouseEvent> ]>;
  private readonly mouseOverAction: PhetioAction<[ Vector2, EventContext<MouseEvent> ]>;
  private readonly mouseOutAction: PhetioAction<[ Vector2, EventContext<MouseEvent> ]>;
  private readonly wheelScrollAction: PhetioAction<[ EventContext<WheelEvent> ]>;
  private readonly touchStartAction: PhetioAction<[ number, Vector2, EventContext<TouchEvent | PointerEvent> ]>;
  private readonly touchEndAction: PhetioAction<[ number, Vector2, EventContext<TouchEvent | PointerEvent> ]>;
  private readonly touchMoveAction: PhetioAction<[ number, Vector2, EventContext<TouchEvent | PointerEvent> ]>;
  private readonly touchCancelAction: PhetioAction<[ number, Vector2, EventContext<TouchEvent | PointerEvent> ]>;
  private readonly penStartAction: PhetioAction<[ number, Vector2, EventContext<PointerEvent> ]>;
  private readonly penEndAction: PhetioAction<[ number, Vector2, EventContext<PointerEvent> ]>;
  private readonly penMoveAction: PhetioAction<[ number, Vector2, EventContext<PointerEvent> ]>;
  private readonly penCancelAction: PhetioAction<[ number, Vector2, EventContext<PointerEvent> ]>;
  private readonly gotPointerCaptureAction: PhetioAction<[ number, EventContext ]>;
  private readonly lostPointerCaptureAction: PhetioAction<[ number, EventContext ]>;

  // If accessible
  private readonly focusinAction: PhetioAction<[ EventContext<FocusEvent> ]>;
  private readonly focusoutAction: PhetioAction<[ EventContext<FocusEvent> ]>;
  private readonly clickAction: PhetioAction<[ EventContext<MouseEvent> ]>;
  private readonly inputAction: PhetioAction<[ EventContext<Event | InputEvent> ]>;
  private readonly changeAction: PhetioAction<[ EventContext ]>;
  private readonly keydownAction: PhetioAction<[ EventContext<KeyboardEvent> ]>;
  private readonly keyupAction: PhetioAction<[ EventContext<KeyboardEvent> ]>;

  public static readonly InputIO = new IOType<Input, InputState>( 'InputIO', {
    valueType: Input,
    applyState: _.noop,
    toStateObject: ( input: Input ) => {
      return {
        pointers: ArrayIOPointerIO.toStateObject( input.pointers )
      };
    },
    stateSchema: {
      pointers: ArrayIOPointerIO
    }
  } );

  /**
   * @param display
   * @param attachToWindow - Whether to add listeners to the window (instead of the Display's domElement).
   * @param batchDOMEvents - If true, most event types will be batched until otherwise triggered.
   * @param assumeFullWindow - We can optimize certain things like computing points if we know the display
   *                                     fills the entire window.
   * @param passiveEvents - See Display's documentation (controls the presence of the passive flag for
   *                                       events, which has some advanced considerations).
   *
   * @param [providedOptions]
   */
  public constructor(
    display: Display,
    attachToWindow: boolean,
    batchDOMEvents: boolean,
    assumeFullWindow: boolean,
    passiveEvents: boolean | null,
    preventMultitouch: boolean,
    interruptMultitouch: boolean,
    providedOptions?: InputOptions
  ) {

    const options = optionize<InputOptions, SelfOptions, PhetioObjectOptions>()( {
      phetioType: Input.InputIO,
      phetioDocumentation: 'Central point for user input events, such as mouse, touch'
    }, providedOptions );

    super( options );

    this.display = display;
    this.rootNode = display.rootNode;

    this.attachToWindow = attachToWindow;
    this.batchDOMEvents = batchDOMEvents;
    this.assumeFullWindow = assumeFullWindow;
    this.passiveEvents = passiveEvents;
    this.preventMultitouch = preventMultitouch;
    this.interruptMultitouch = interruptMultitouch;
    this.batchedEvents = [];
    this.pdomPointer = null;
    this.mouse = null;
    this.pointers = [];
    this.pointerAddedEmitter = new TinyEmitter<[ Pointer ]>();
    this.currentlyFiringEvents = false;
    this.upTimeStamp = 0;

    ////////////////////////////////////////////////////
    // Declare the Actions that send scenery input events to the PhET-iO data stream.  Note they use the default value
    // of phetioReadOnly false, in case a client wants to synthesize events.

    this.validatePointersAction = new PhetioAction( () => {
      let i = this.pointers.length;
      while ( i-- ) {
        const pointer = this.pointers[ i ];
        if ( pointer.point && pointer !== this.pdomPointer ) {
          this.branchChangeEvents<Event>( pointer, pointer.lastEventContext || EventContext.createSynthetic(), false );
        }
      }
    }, {
      phetioPlayback: true,
      tandem: options.tandem?.createTandem( 'validatePointersAction' ),
      phetioHighFrequency: true
    } );

    this.mouseUpAction = new PhetioAction( ( point: Vector2, context: EventContext<MouseEvent> ) => {
      const mouse = this.ensureMouse( point, context );
      mouse.id = null;
      this.upEvent<MouseEvent>( mouse, context, point );
    }, {
      phetioPlayback: true,
      tandem: options.tandem?.createTandem( 'mouseUpAction' ),
      parameters: [
        { name: 'point', phetioType: Vector2.Vector2IO },
        { name: 'context', phetioType: EventContextIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when a mouse button is released.'
    } );

    this.mouseDownAction = new PhetioAction( ( id: number, point: Vector2, context: EventContext<MouseEvent> ) => {
      if ( this.interruptMultitouch ) {
        this.interruptMultitouchPointers( point, context );
      }

      const mouse = this.ensureMouse( point, context );
      mouse.id = id;
      this.downEvent<MouseEvent>( mouse, context, point );
    }, {
      phetioPlayback: true,
      tandem: options.tandem?.createTandem( 'mouseDownAction' ),
      parameters: [
        { name: 'id', phetioType: NullableIO( NumberIO ) },
        { name: 'point', phetioType: Vector2.Vector2IO },
        { name: 'context', phetioType: EventContextIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when a mouse button is pressed.'
    } );

    this.mouseMoveAction = new PhetioAction( ( point: Vector2, context: EventContext<MouseEvent> ) => {
      const mouse = this.ensureMouse( point, context );
      mouse.move( point );
      this.moveEvent<MouseEvent>( mouse, context );
    }, {
      phetioPlayback: true,
      tandem: options.tandem?.createTandem( 'mouseMoveAction' ),
      parameters: [
        { name: 'point', phetioType: Vector2.Vector2IO },
        { name: 'context', phetioType: EventContextIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when the mouse is moved.',
      phetioHighFrequency: true
    } );

    this.mouseOverAction = new PhetioAction( ( point: Vector2, context: EventContext<MouseEvent> ) => {
      const mouse = this.ensureMouse( point, context );
      mouse.over( point );
      // TODO: how to handle mouse-over (and log it)... are we changing the pointer.point without a branch change? https://github.com/phetsims/scenery/issues/1581
    }, {
      phetioPlayback: true,
      tandem: options.tandem?.createTandem( 'mouseOverAction' ),
      parameters: [
        { name: 'point', phetioType: Vector2.Vector2IO },
        { name: 'context', phetioType: EventContextIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when the mouse is moved while on the sim.'
    } );

    this.mouseOutAction = new PhetioAction( ( point: Vector2, context: EventContext<MouseEvent> ) => {
      const mouse = this.ensureMouse( point, context );
      mouse.out( point );
      // TODO: how to handle mouse-out (and log it)... are we changing the pointer.point without a branch change? https://github.com/phetsims/scenery/issues/1581
    }, {
      phetioPlayback: true,
      tandem: options.tandem?.createTandem( 'mouseOutAction' ),
      parameters: [
        { name: 'point', phetioType: Vector2.Vector2IO },
        { name: 'context', phetioType: EventContextIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when the mouse moves out of the display.'
    } );

    this.wheelScrollAction = new PhetioAction( ( context: EventContext<WheelEvent> ) => {
      const event = context.domEvent;

      const mouse = this.ensureMouse( this.pointFromEvent( event ), context );
      mouse.wheel( event );

      // don't send mouse-wheel events if we don't yet have a mouse location!
      // TODO: Can we set the mouse location based on the wheel event? https://github.com/phetsims/scenery/issues/1581
      if ( mouse.point ) {
        const trail = this.rootNode.trailUnderPointer( mouse ) || new Trail( this.rootNode );
        this.dispatchEvent<WheelEvent>( trail, 'wheel', mouse, context, true );
      }
    }, {
      phetioPlayback: true,
      tandem: options.tandem?.createTandem( 'wheelScrollAction' ),
      parameters: [
        { name: 'context', phetioType: EventContextIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when the mouse wheel scrolls.',
      phetioHighFrequency: true
    } );

    this.touchStartAction = new PhetioAction( ( id: number, point: Vector2, context: EventContext<TouchEvent | PointerEvent> ) => {
      const touch = new Touch( id, point, context.domEvent );
      this.addPointer( touch, point, context );
      this.downEvent<TouchEvent | PointerEvent>( touch, context, point );
    }, {
      phetioPlayback: true,
      tandem: options.tandem?.createTandem( 'touchStartAction' ),
      parameters: [
        { name: 'id', phetioType: NumberIO },
        { name: 'point', phetioType: Vector2.Vector2IO },
        { name: 'context', phetioType: EventContextIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when a touch begins.'
    } );

    this.touchEndAction = new PhetioAction( ( id: number, point: Vector2, context: EventContext<TouchEvent | PointerEvent> ) => {
      const touch = this.findPointerById( id ) as Touch | null;
      if ( touch ) {
        assert && assert( touch instanceof Touch ); // eslint-disable-line phet/no-simple-type-checking-assertions, phet/bad-sim-text
        this.upEvent<TouchEvent | PointerEvent>( touch, context, point );
        this.removePointer( touch );
      }
    }, {
      phetioPlayback: true,
      tandem: options.tandem?.createTandem( 'touchEndAction' ),
      parameters: [
        { name: 'id', phetioType: NumberIO },
        { name: 'point', phetioType: Vector2.Vector2IO },
        { name: 'context', phetioType: EventContextIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when a touch ends.'
    } );

    this.touchMoveAction = new PhetioAction( ( id: number, point: Vector2, context: EventContext<TouchEvent | PointerEvent> ) => {
      const touch = this.findPointerById( id ) as Touch | null;
      if ( touch ) {
        assert && assert( touch instanceof Touch ); // eslint-disable-line phet/no-simple-type-checking-assertions, phet/bad-sim-text
        touch.move( point );
        this.moveEvent<TouchEvent | PointerEvent>( touch, context );
      }
    }, {
      phetioPlayback: true,
      tandem: options.tandem?.createTandem( 'touchMoveAction' ),
      parameters: [
        { name: 'id', phetioType: NumberIO },
        { name: 'point', phetioType: Vector2.Vector2IO },
        { name: 'context', phetioType: EventContextIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when a touch moves.',
      phetioHighFrequency: true
    } );

    this.touchCancelAction = new PhetioAction( ( id: number, point: Vector2, context: EventContext<TouchEvent | PointerEvent> ) => {
      const touch = this.findPointerById( id ) as Touch | null;
      if ( touch ) {
        assert && assert( touch instanceof Touch ); // eslint-disable-line phet/no-simple-type-checking-assertions, phet/bad-sim-text
        this.cancelEvent<TouchEvent | PointerEvent>( touch, context, point );
        this.removePointer( touch );
      }
    }, {
      phetioPlayback: true,
      tandem: options.tandem?.createTandem( 'touchCancelAction' ),
      parameters: [
        { name: 'id', phetioType: NumberIO },
        { name: 'point', phetioType: Vector2.Vector2IO },
        { name: 'context', phetioType: EventContextIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when a touch is canceled.'
    } );

    this.penStartAction = new PhetioAction( ( id: number, point: Vector2, context: EventContext<PointerEvent> ) => {
      const pen = new Pen( id, point, context.domEvent );
      this.addPointer( pen, point, context );
      this.downEvent<PointerEvent>( pen, context, point );
    }, {
      phetioPlayback: true,
      tandem: options.tandem?.createTandem( 'penStartAction' ),
      parameters: [
        { name: 'id', phetioType: NumberIO },
        { name: 'point', phetioType: Vector2.Vector2IO },
        { name: 'context', phetioType: EventContextIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when a pen touches the screen.'
    } );

    this.penEndAction = new PhetioAction( ( id: number, point: Vector2, context: EventContext<PointerEvent> ) => {
      const pen = this.findPointerById( id ) as Pen | null;
      if ( pen ) {
        this.upEvent<PointerEvent>( pen, context, point );
        this.removePointer( pen );
      }
    }, {
      phetioPlayback: true,
      tandem: options.tandem?.createTandem( 'penEndAction' ),
      parameters: [
        { name: 'id', phetioType: NumberIO },
        { name: 'point', phetioType: Vector2.Vector2IO },
        { name: 'context', phetioType: EventContextIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when a pen is lifted.'
    } );

    this.penMoveAction = new PhetioAction( ( id: number, point: Vector2, context: EventContext<PointerEvent> ) => {
      const pen = this.findPointerById( id ) as Pen | null;
      if ( pen ) {
        pen.move( point );
        this.moveEvent<PointerEvent>( pen, context );
      }
    }, {
      phetioPlayback: true,
      tandem: options.tandem?.createTandem( 'penMoveAction' ),
      parameters: [
        { name: 'id', phetioType: NumberIO },
        { name: 'point', phetioType: Vector2.Vector2IO },
        { name: 'context', phetioType: EventContextIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when a pen is moved.',
      phetioHighFrequency: true
    } );

    this.penCancelAction = new PhetioAction( ( id: number, point: Vector2, context: EventContext<PointerEvent> ) => {
      const pen = this.findPointerById( id ) as Pen | null;
      if ( pen ) {
        this.cancelEvent<PointerEvent>( pen, context, point );
        this.removePointer( pen );
      }
    }, {
      phetioPlayback: true,
      tandem: options.tandem?.createTandem( 'penCancelAction' ),
      parameters: [
        { name: 'id', phetioType: NumberIO },
        { name: 'point', phetioType: Vector2.Vector2IO },
        { name: 'context', phetioType: EventContextIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when a pen is canceled.'
    } );

    this.gotPointerCaptureAction = new PhetioAction( ( id: number, context: EventContext ) => {
      const pointer = this.findPointerById( id );

      if ( pointer ) {
        pointer.onGotPointerCapture();
      }
    }, {
      phetioPlayback: true,
      tandem: options.tandem?.createTandem( 'gotPointerCaptureAction' ),
      parameters: [
        { name: 'id', phetioType: NumberIO },
        { name: 'context', phetioType: EventContextIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when a pointer is captured (normally at the start of an interaction)',
      phetioHighFrequency: true
    } );

    this.lostPointerCaptureAction = new PhetioAction( ( id: number, context: EventContext ) => {
      const pointer = this.findPointerById( id );

      if ( pointer ) {

        // While investigating https://github.com/phetsims/mean-share-and-balance/issues/336 it was discovered that
        // pointerUp events were not being transmitted before lostPointerCapture events by the browser. This was
        // surprising since the spec says otherwise: https://w3c.github.io/pointerevents/#implicit-release-of-pointer-capture
        // This behavior was found to be occurring while using a mac trackpad with finger movements that could
        // potentially be perceived as gesture. The following setTimeout gives the pointerUp event a chance to fire first
        // before the lostPointerCapture event so that listeners can rely on the expected order.
        stepTimer.setTimeout( () => { pointer.onLostPointerCapture(); }, 2 );
      }
    }, {
      phetioPlayback: true,
      tandem: options.tandem?.createTandem( 'lostPointerCaptureAction' ),
      parameters: [
        { name: 'id', phetioType: NumberIO },
        { name: 'context', phetioType: EventContextIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when a pointer loses its capture (normally at the end of an interaction)',
      phetioHighFrequency: true
    } );

    this.focusinAction = new PhetioAction( ( context: EventContext<FocusEvent> ) => {
      const trail = this.getPDOMEventTrail( context.domEvent, 'focusin' );
      if ( !trail ) {
        return;
      }

      sceneryLog && sceneryLog.Input && sceneryLog.Input( `focusin(${Input.debugText( null, context.domEvent )});` );
      sceneryLog && sceneryLog.Input && sceneryLog.push();

      this.dispatchPDOMEvent<FocusEvent>( trail, 'focus', context, false );
      this.dispatchPDOMEvent<FocusEvent>( trail, 'focusin', context, true );

      sceneryLog && sceneryLog.Input && sceneryLog.pop();
    }, {
      phetioPlayback: true,
      tandem: options.tandem?.createTandem( 'focusinAction' ),
      parameters: [
        { name: 'context', phetioType: EventContextIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when the PDOM root gets the focusin DOM event.'
    } );

    this.focusoutAction = new PhetioAction( ( context: EventContext<FocusEvent> ) => {
      const trail = this.getPDOMEventTrail( context.domEvent, 'focusout' );
      if ( !trail ) {
        return;
      }

      sceneryLog && sceneryLog.Input && sceneryLog.Input( `focusOut(${Input.debugText( null, context.domEvent )});` );
      sceneryLog && sceneryLog.Input && sceneryLog.push();

      this.dispatchPDOMEvent<FocusEvent>( trail, 'blur', context, false );
      this.dispatchPDOMEvent<FocusEvent>( trail, 'focusout', context, true );

      sceneryLog && sceneryLog.Input && sceneryLog.pop();
    }, {
      phetioPlayback: true,
      tandem: options.tandem?.createTandem( 'focusoutAction' ),
      parameters: [
        { name: 'context', phetioType: EventContextIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when the PDOM root gets the focusout DOM event.'
    } );

    // https://developer.mozilla.org/en-US/docs/Web/API/Element/click_event notes that the click action should result
    // in a MouseEvent
    this.clickAction = new PhetioAction( ( context: EventContext<MouseEvent> ) => {
      const trail = this.getPDOMEventTrail( context.domEvent, 'click' );
      if ( !trail ) {
        return;
      }

      sceneryLog && sceneryLog.Input && sceneryLog.Input( `click(${Input.debugText( null, context.domEvent )});` );
      sceneryLog && sceneryLog.Input && sceneryLog.push();

      this.dispatchPDOMEvent<MouseEvent>( trail, 'click', context, true );

      sceneryLog && sceneryLog.Input && sceneryLog.pop();
    }, {
      phetioPlayback: true,
      tandem: options.tandem?.createTandem( 'clickAction' ),
      parameters: [
        { name: 'context', phetioType: EventContextIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when the PDOM root gets the click DOM event.'
    } );

    this.inputAction = new PhetioAction( ( context: EventContext<Event | InputEvent> ) => {
      const trail = this.getPDOMEventTrail( context.domEvent, 'input' );
      if ( !trail ) {
        return;
      }

      sceneryLog && sceneryLog.Input && sceneryLog.Input( `input(${Input.debugText( null, context.domEvent )});` );
      sceneryLog && sceneryLog.Input && sceneryLog.push();

      this.dispatchPDOMEvent<Event | InputEvent>( trail, 'input', context, true );

      sceneryLog && sceneryLog.Input && sceneryLog.pop();
    }, {
      phetioPlayback: true,
      tandem: options.tandem?.createTandem( 'inputAction' ),
      parameters: [
        { name: 'context', phetioType: EventContextIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when the PDOM root gets the input DOM event.'
    } );

    this.changeAction = new PhetioAction( ( context: EventContext ) => {
      const trail = this.getPDOMEventTrail( context.domEvent, 'change' );
      if ( !trail ) {
        return;
      }

      sceneryLog && sceneryLog.Input && sceneryLog.Input( `change(${Input.debugText( null, context.domEvent )});` );
      sceneryLog && sceneryLog.Input && sceneryLog.push();

      this.dispatchPDOMEvent<Event>( trail, 'change', context, true );

      sceneryLog && sceneryLog.Input && sceneryLog.pop();
    }, {
      phetioPlayback: true,
      tandem: options.tandem?.createTandem( 'changeAction' ),
      parameters: [
        { name: 'context', phetioType: EventContextIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when the PDOM root gets the change DOM event.'
    } );

    this.keydownAction = new PhetioAction( ( context: EventContext<KeyboardEvent> ) => {
      sceneryLog && sceneryLog.Input && sceneryLog.Input( `keydown(${Input.debugText( null, context.domEvent )});` );
      sceneryLog && sceneryLog.Input && sceneryLog.push();

      const trail = this.getPDOMEventTrail( context.domEvent, 'keydown' );
      trail && this.dispatchPDOMEvent<KeyboardEvent>( trail, 'keydown', context, true );

      sceneryLog && sceneryLog.Input && sceneryLog.pop();
    }, {
      phetioPlayback: true,
      tandem: options.tandem?.createTandem( 'keydownAction' ),
      parameters: [
        { name: 'context', phetioType: EventContextIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when the PDOM root gets the keydown DOM event.'
    } );

    this.keyupAction = new PhetioAction( ( context: EventContext<KeyboardEvent> ) => {
      sceneryLog && sceneryLog.Input && sceneryLog.Input( `keyup(${Input.debugText( null, context.domEvent )});` );
      sceneryLog && sceneryLog.Input && sceneryLog.push();

      const trail = this.getPDOMEventTrail( context.domEvent, 'keydown' );
      trail && this.dispatchPDOMEvent<KeyboardEvent>( trail, 'keyup', context, true );

      sceneryLog && sceneryLog.Input && sceneryLog.pop();
    }, {
      phetioPlayback: true,
      tandem: options.tandem?.createTandem( 'keyupAction' ),
      parameters: [
        { name: 'context', phetioType: EventContextIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when the PDOM root gets the keyup DOM event.'
    } );
  }

  /**
   * Interrupts any input actions that are currently taking place (should stop drags, etc.)
   *
   * If excludePointer is provided, it will NOT be interrupted along with the others
   */
  public interruptPointers( excludePointer: Pointer | null = null ): void {
    _.each( this.pointers, pointer => {
      if ( pointer !== excludePointer ) {
        pointer.interruptAll();
      }
    } );
  }

  /**
   * Called to batch a raw DOM event (which may be immediately fired, depending on the settings). (scenery-internal)
   *
   * @param context
   * @param batchType - See BatchedDOMEvent's "enumeration"
   * @param callback - Parameter types defined by the batchType. See BatchedDOMEvent for details
   * @param triggerImmediate - Certain events can force immediate action, since browsers like Chrome
   *                                     only allow certain operations in the callback for a user gesture (e.g. like
   *                                     a mouseup to open a window).
   */
  public batchEvent( context: EventContext, batchType: BatchedDOMEventType, callback: BatchedDOMEventCallback, triggerImmediate: boolean ): void {
    sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent( 'Input.batchEvent' );
    sceneryLog && sceneryLog.InputEvent && sceneryLog.push();

    // If our display is not interactive, do not respond to any events (but still prevent default)
    if ( this.display.interactive ) {
      this.batchedEvents.push( BatchedDOMEvent.pool.create( context, batchType, callback ) );
      if ( triggerImmediate || !this.batchDOMEvents ) {
        this.fireBatchedEvents();
      }
      // NOTE: If we ever want to Display.updateDisplay() on events, do so here
    }

    // Firefox has issues with preventDefault on pointer down events +
    // multitouch. https://github.com/scenerystack/scenerystack/issues/42 tracks
    // the main issue, we skip preventDefault here to prevent
    // https://github.com/phetsims/scenery/issues/837 from happening.
    // https://bugzilla.mozilla.org/show_bug.cgi?id=1729465 notes preventDefault()
    // as the culprit.
    const isFirefoxPointerDownTouch =
      platform.firefox &&
      context.domEvent instanceof PointerEvent &&
      callback === this.pointerDown &&
      context.domEvent.pointerType === 'touch';

    // Always preventDefault on touch events, since we don't want mouse events triggered afterwards. See
    // http://www.html5rocks.com/en/mobile/touchandmouse/ for more information.
    // Additionally, IE had some issues with skipping prevent default, see
    // https://github.com/phetsims/scenery/issues/464 for mouse handling.
    // WE WILL NOT preventDefault() on keyboard or alternative input events here
    if (
      !( this.passiveEvents === true ) &&
      ( callback !== this.mouseDown || platform.edge ) &&
      !isFirefoxPointerDownTouch &&
      batchType !== BatchedDOMEventType.ALT_TYPE &&
      !context.allowsDOMInput()
    ) {
      // We cannot prevent a passive event, so don't try
      context.domEvent.preventDefault();
    }

    sceneryLog && sceneryLog.InputEvent && sceneryLog.pop();
  }

  /**
   * Fires all of our events that were batched into the batchedEvents array. (scenery-internal)
   */
  public fireBatchedEvents(): void {
    sceneryLog && sceneryLog.InputEvent && this.currentlyFiringEvents && sceneryLog.InputEvent(
      'REENTRANCE DETECTED' );
    // Don't re-entrantly enter our loop, see https://github.com/phetsims/balloons-and-static-electricity/issues/406
    if ( !this.currentlyFiringEvents && this.batchedEvents.length ) {
      sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent( `Input.fireBatchedEvents length:${this.batchedEvents.length}` );
      sceneryLog && sceneryLog.InputEvent && sceneryLog.push();

      this.currentlyFiringEvents = true;

      // needs to be done in order
      const batchedEvents = this.batchedEvents;
      // IMPORTANT: We need to check the length of the array at every iteration, as it can change due to re-entrant
      // event handling, see https://github.com/phetsims/balloons-and-static-electricity/issues/406.
      // Events may be appended to this (synchronously) as part of firing initial events, so we want to FULLY run all
      // events before clearing our array.
      for ( let i = 0; i < batchedEvents.length; i++ ) {
        const batchedEvent = batchedEvents[ i ];
        batchedEvent.run( this );
        batchedEvent.dispose();
      }
      cleanArray( batchedEvents );

      this.currentlyFiringEvents = false;

      sceneryLog && sceneryLog.InputEvent && sceneryLog.pop();
    }
  }

  /**
   * Clears any batched events that we don't want to process. (scenery-internal)
   *
   * NOTE: It is HIGHLY recommended to interrupt pointers and remove non-Mouse pointers before doing this, as
   * otherwise it can cause incorrect state in certain types of listeners (e.g. ones that count how many pointers
   * are over them).
   */
  public clearBatchedEvents(): void {
    this.batchedEvents.length = 0;
  }

  /**
   * Checks all pointers to see whether they are still "over" the same nodes (trail). If not, it will fire the usual
   * enter/exit events. (scenery-internal)
   */
  public validatePointers(): void {
    this.validatePointersAction.execute();
  }

  /**
   * Removes all non-Mouse pointers from internal tracking. (scenery-internal)
   */
  public removeTemporaryPointers(): void {
    for ( let i = this.pointers.length - 1; i >= 0; i-- ) {
      const pointer = this.pointers[ i ];
      if ( !( pointer instanceof Mouse ) ) {
        this.pointers.splice( i, 1 );

        // Send exit events. As we can't get a DOM event, we'll send a fake object instead.
        const exitTrail = pointer.trail || new Trail( this.rootNode );
        this.exitEvents( pointer, EventContext.createSynthetic(), exitTrail, 0, true );
      }
    }
  }

  /**
   * Hooks up DOM listeners to whatever type of object we are going to listen to. (scenery-internal)
   */
  public connectListeners(): void {
    BrowserEvents.addDisplay( this.display, this.attachToWindow, this.passiveEvents );
  }

  /**
   * Removes DOM listeners from whatever type of object we were listening to. (scenery-internal)
   */
  public disconnectListeners(): void {
    BrowserEvents.removeDisplay( this.display, this.attachToWindow, this.passiveEvents );
  }

  /**
   * Extract a {Vector2} global coordinate point from an arbitrary DOM event. (scenery-internal)
   */
  public pointFromEvent( domEvent: MouseEvent | WindowTouch ): Vector2 {
    const position = Vector2.pool.create( domEvent.clientX, domEvent.clientY );
    if ( !this.assumeFullWindow ) {
      const domBounds = this.display.domElement.getBoundingClientRect();

      // TODO: consider totally ignoring any with zero width/height, as we aren't attached to the display? https://github.com/phetsims/scenery/issues/1581
      // For now, don't offset.
      if ( domBounds.width > 0 && domBounds.height > 0 ) {
        position.subtractXY( domBounds.left, domBounds.top );

        // Detect a scaling of the display here (the client bounding rect having different dimensions from our
        // display), and attempt to compensate.
        // NOTE: We can't handle rotation here.
        if ( domBounds.width !== this.display.width || domBounds.height !== this.display.height ) {
          // TODO: Have code verify the correctness here, and that it's not triggering all the time https://github.com/phetsims/scenery/issues/1581
          position.x *= this.display.width / domBounds.width;
          position.y *= this.display.height / domBounds.height;
        }
      }
    }
    return position;
  }

  private interruptMultitouchPointers( point: Vector2, context: EventContext ): void {
    const touchLikePointers = this.pointers.filter( otherPointer => otherPointer.isTouchLike() );
    for ( const otherPointer of touchLikePointers ) {
      this.cancelEvent( otherPointer, context, point );
      this.removePointer( otherPointer );
    }
  }

  /**
   * Adds a pointer to our list.
   */
  private addPointer( pointer: Pointer, point: Vector2, context: EventContext ): void {
    // Disable multitouch, see https://github.com/phetsims/scenery/issues/1684
    // For now, this is done by preventing a new "pointer" from being added, so that
    // the events that would be forwarded to it are ignored.
    // NOTE: NOT doing ` || this.mouse?.hasPointerCaptured()`, since it seems
    // like we are not fully clearing the capture state when we should be.
    if (
      this.preventMultitouch &&
      pointer.isTouchLike() &&
      this.pointers.some( otherPointer => otherPointer.isTouchLike() )
    ) {
      return;
    }

    // Interrupt touchlike pointers on a new pointer
    if ( this.interruptMultitouch ) {
      this.interruptMultitouchPointers( point, context );

      // And if we have a mouse, interrupt listeners on it as well.
      if ( this.mouse ) {
        this.mouse.interruptAll();
      }
    }

    this.pointers.push( pointer );

    this.pointerAddedEmitter.emit( pointer );
  }

  /**
   * Removes a pointer from our list. If we get future events for it (based on the ID) it will be ignored.
   */
  private removePointer( pointer: Pointer ): void {
    // sanity check version, will remove all instances
    for ( let i = this.pointers.length - 1; i >= 0; i-- ) {
      if ( this.pointers[ i ] === pointer ) {
        this.pointers.splice( i, 1 );
      }
    }

    pointer.dispose();
  }

  /**
   * Given a pointer's ID (given by the pointer/touch specifications to be unique to a specific pointer/touch),
   * returns the given pointer (if we have one).
   *
   * NOTE: There are some cases where we may have prematurely "removed" a pointer.
   */
  private findPointerById( id: number ): Mouse | Touch | Pen | null {
    let i = this.pointers.length;
    while ( i-- ) {
      const pointer = this.pointers[ i ] as Mouse | Touch | Pen;
      if ( pointer.id === id ) {
        return pointer;
      }
    }
    return null;
  }

  private getPDOMEventTrail( domEvent: TargetSubstitudeAugmentedEvent, eventName: string ): Trail | null {
    if ( !this.display.interactive ) {
      return null;
    }

    const trail = this.getTrailFromPDOMEvent( domEvent );

    // Only dispatch the event if the click did not happen rapidly after an up event. It is
    // likely that the screen reader dispatched both pointer AND click events in this case, and
    // we only want to respond to one or the other. See https://github.com/phetsims/scenery/issues/1094.
    // This is outside of the clickAction execution so that blocked clicks are not part of the PhET-iO data
    // stream.
    const notBlockingSubsequentClicksOccurringTooQuickly = trail && !( eventName === 'click' &&
                                                           _.some( trail.nodes, node => node.positionInPDOM ) &&
                                                           domEvent.timeStamp - this.upTimeStamp <= PDOM_CLICK_DELAY );

    return notBlockingSubsequentClicksOccurringTooQuickly ? trail : null;
  }

  private ensureMouse( point: Vector2, context: EventContext<MouseEvent> | EventContext<WheelEvent> ): Mouse {
    const mouse = this.mouse;
    if ( mouse ) {
      return mouse;
    }
    else {
      // Initializes the Mouse object on the first mouse event (this may never happen on touch devices).
      const mouse = new Mouse( point );
      this.mouse = mouse;
      this.addPointer( mouse, point, context );
      return mouse;
    }
  }

  private ensurePDOMPointer( context: EventContext ): PDOMPointer {
    const pdomPointer = this.pdomPointer;
    if ( pdomPointer ) {
      return pdomPointer;
    }
    else {
      // Initializes the accessible pointer object on the first pdom event.
      const pdomPointer = new PDOMPointer( this.display );
      this.pdomPointer = pdomPointer;

      this.addPointer( pdomPointer, Vector2.ZERO, context );

      return pdomPointer;
    }
  }

  /**
   * Steps to dispatch a pdom-related event. Before dispatch, the PDOMPointer is initialized if it
   * hasn't been created yet and a userGestureEmitter emits to indicate that a user has begun an interaction.
   */
  private dispatchPDOMEvent<DOMEvent extends Event>( trail: Trail, eventType: SupportedEventTypes, context: EventContext<DOMEvent>, bubbles: boolean ): void {

    this.ensurePDOMPointer( context ).updateTrail( trail );

    // exclude focus and blur events because they can happen with scripting without user input
    if ( PDOMUtils.USER_GESTURE_EVENTS.includes( eventType ) ) {
      DisplayGlobals.userGestureEmitter.emit();
    }

    const domEvent = context.domEvent;

    // This workaround hopefully won't be here forever, see ParallelDOM.setExcludeLabelSiblingFromInput() and https://github.com/phetsims/a11y-research/issues/156
    if ( !( domEvent.target && ( domEvent.target as Element ).hasAttribute( PDOMUtils.DATA_EXCLUDE_FROM_INPUT ) ) ) {

      // If the trail is not pickable, do not dispatch to those targets. However, a subset of events WILL still dispatch
      // to targets to support focus events on disabled/non-interactive Nodes.
      const canDispatchToTrail = trail.isPickable() || PDOM_ALWAYS_ACTIVE_EVENTS.includes( eventType );

      assert && assert( this.pdomPointer );
      this.dispatchEvent<DOMEvent>( trail, eventType, this.pdomPointer!, context, bubbles, PDOM_ALWAYS_ACTIVE_EVENTS.includes( eventType ), canDispatchToTrail );
    }
  }

  /**
   * From a DOM Event, get its relatedTarget and map that to the scenery Node. Will return null if relatedTarget
   * is not provided, or if relatedTarget is not under PDOM, or there is no associated Node with trail id on the
   * relatedTarget element. (scenery-internal)
   *
   * @param domEvent - DOM Event, not a SceneryEvent!
   */
  public getRelatedTargetTrail( domEvent: FocusEvent | MouseEvent ): Trail | null {
    const relatedTargetElement = domEvent.relatedTarget;

    if ( relatedTargetElement && this.display.isElementUnderPDOM( relatedTargetElement as HTMLElement, false ) ) {

      const relatedTarget = ( domEvent.relatedTarget as unknown as Element );
      assert && assert( relatedTarget instanceof window.Element );
      const trailIndices = relatedTarget.getAttribute( PDOMUtils.DATA_PDOM_UNIQUE_ID );
      assert && assert( trailIndices, 'should not be null' );

      return pdomUniqueIdToTrail( this.display, trailIndices! );
    }
    return null;
  }

  /**
   * Get the trail ID of the node represented by a DOM element who is the target of a DOM Event in the accessible PDOM.
   * This is a bit of a misnomer, because the domEvent doesn't have to be under the PDOM. Returns null if not in the PDOM.
   */
  private getTrailFromPDOMEvent( domEvent: TargetSubstitudeAugmentedEvent ): Trail | null {
    assert && assert( domEvent.target || domEvent[ TARGET_SUBSTITUTE_KEY ], 'need a way to get the target' );

    if ( !this.display._accessible ) {
      return null;
    }

    // could be serialized event for phet-io playbacks, see Input.serializeDOMEvent()
    if ( domEvent[ TARGET_SUBSTITUTE_KEY ] ) {
      const trailIndices = domEvent[ TARGET_SUBSTITUTE_KEY ].getAttribute( PDOMUtils.DATA_PDOM_UNIQUE_ID );
      return pdomUniqueIdToTrail( this.display, trailIndices! );
    }
    else {
      const target = domEvent.target;
      if ( target && target instanceof window.Element && this.display.isElementUnderPDOM( target, false ) ) {
        const trailIndices = target.getAttribute( PDOMUtils.DATA_PDOM_UNIQUE_ID );
        assert && assert( trailIndices, 'should not be null' );
        return pdomUniqueIdToTrail( this.display, trailIndices! );
      }
    }
    return null;
  }

  /**
   * Triggers a logical mousedown event. (scenery-internal)
   *
   * NOTE: This may also be called from the pointer event handler (pointerDown) or from things like fuzzing or
   * playback. The event may be "faked" for certain purposes.
   */
  public mouseDown( id: number, point: Vector2, context: EventContext<MouseEvent | PointerEvent> ): void {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `mouseDown('${id}', ${Input.debugText( point, context.domEvent )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.mouseDownAction.execute( id, point, context );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical mouseup event. (scenery-internal)
   *
   * NOTE: This may also be called from the pointer event handler (pointerUp) or from things like fuzzing or
   * playback. The event may be "faked" for certain purposes.
   */
  public mouseUp( point: Vector2, context: EventContext<MouseEvent | PointerEvent> ): void {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `mouseUp(${Input.debugText( point, context.domEvent )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.mouseUpAction.execute( point, context );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical mousemove event. (scenery-internal)
   *
   * NOTE: This may also be called from the pointer event handler (pointerMove) or from things like fuzzing or
   * playback. The event may be "faked" for certain purposes.
   */
  public mouseMove( point: Vector2, context: EventContext<MouseEvent | PointerEvent> ): void {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `mouseMove(${Input.debugText( point, context.domEvent )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.mouseMoveAction.execute( point, context );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical mouseover event (this does NOT correspond to the Scenery event, since this is for the display) (scenery-internal)
   */
  public mouseOver( point: Vector2, context: EventContext<MouseEvent | PointerEvent> ): void {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `mouseOver(${Input.debugText( point, context.domEvent )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.mouseOverAction.execute( point, context );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical mouseout event (this does NOT correspond to the Scenery event, since this is for the display) (scenery-internal)
   */
  public mouseOut( point: Vector2, context: EventContext<MouseEvent | PointerEvent> ): void {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `mouseOut(${Input.debugText( point, context.domEvent )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.mouseOutAction.execute( point, context );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical mouse-wheel/scroll event. (scenery-internal)
   */
  public wheel( context: EventContext<WheelEvent> ): void {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `wheel(${Input.debugText( null, context.domEvent )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.wheelScrollAction.execute( context );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical touchstart event. This is called for each touch point in a 'raw' event. (scenery-internal)
   *
   * NOTE: This may also be called from the pointer event handler (pointerDown) or from things like fuzzing or
   * playback. The event may be "faked" for certain purposes.
   */
  public touchStart( id: number, point: Vector2, context: EventContext<TouchEvent | PointerEvent> ): void {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `touchStart('${id}',${Input.debugText( point, context.domEvent )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();

    this.touchStartAction.execute( id, point, context );

    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical touchend event. This is called for each touch point in a 'raw' event. (scenery-internal)
   *
   * NOTE: This may also be called from the pointer event handler (pointerUp) or from things like fuzzing or
   * playback. The event may be "faked" for certain purposes.
   */
  public touchEnd( id: number, point: Vector2, context: EventContext<TouchEvent | PointerEvent> ): void {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `touchEnd('${id}',${Input.debugText( point, context.domEvent )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();

    this.touchEndAction.execute( id, point, context );

    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical touchmove event. This is called for each touch point in a 'raw' event. (scenery-internal)
   *
   * NOTE: This may also be called from the pointer event handler (pointerMove) or from things like fuzzing or
   * playback. The event may be "faked" for certain purposes.
   */
  public touchMove( id: number, point: Vector2, context: EventContext<TouchEvent | PointerEvent> ): void {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `touchMove('${id}',${Input.debugText( point, context.domEvent )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.touchMoveAction.execute( id, point, context );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical touchcancel event. This is called for each touch point in a 'raw' event. (scenery-internal)
   *
   * NOTE: This may also be called from the pointer event handler (pointerCancel) or from things like fuzzing or
   * playback. The event may be "faked" for certain purposes.
   */
  public touchCancel( id: number, point: Vector2, context: EventContext<TouchEvent | PointerEvent> ): void {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `touchCancel('${id}',${Input.debugText( point, context.domEvent )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.touchCancelAction.execute( id, point, context );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical penstart event (e.g. a stylus). This is called for each pen point in a 'raw' event. (scenery-internal)
   *
   * NOTE: This may also be called from the pointer event handler (pointerDown) or from things like fuzzing or
   * playback. The event may be "faked" for certain purposes.
   */
  public penStart( id: number, point: Vector2, context: EventContext<PointerEvent> ): void {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `penStart('${id}',${Input.debugText( point, context.domEvent )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.penStartAction.execute( id, point, context );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical penend event (e.g. a stylus). This is called for each pen point in a 'raw' event. (scenery-internal)
   *
   * NOTE: This may also be called from the pointer event handler (pointerUp) or from things like fuzzing or
   * playback. The event may be "faked" for certain purposes.
   */
  public penEnd( id: number, point: Vector2, context: EventContext<PointerEvent> ): void {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `penEnd('${id}',${Input.debugText( point, context.domEvent )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.penEndAction.execute( id, point, context );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical penmove event (e.g. a stylus). This is called for each pen point in a 'raw' event. (scenery-internal)
   *
   * NOTE: This may also be called from the pointer event handler (pointerMove) or from things like fuzzing or
   * playback. The event may be "faked" for certain purposes.
   */
  public penMove( id: number, point: Vector2, context: EventContext<PointerEvent> ): void {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `penMove('${id}',${Input.debugText( point, context.domEvent )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.penMoveAction.execute( id, point, context );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical pencancel event (e.g. a stylus). This is called for each pen point in a 'raw' event. (scenery-internal)
   *
   * NOTE: This may also be called from the pointer event handler (pointerCancel) or from things like fuzzing or
   * playback. The event may be "faked" for certain purposes.
   */
  public penCancel( id: number, point: Vector2, context: EventContext<PointerEvent> ): void {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `penCancel('${id}',${Input.debugText( point, context.domEvent )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.penCancelAction.execute( id, point, context );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Handles a pointerdown event, forwarding it to the proper logical event. (scenery-internal)
   */
  public pointerDown( id: number, type: string, point: Vector2, context: EventContext<PointerEvent> ): void {
    // In IE for pointer down events, we want to make sure than the next interactions off the page are sent to
    // this element (it will bubble). See https://github.com/phetsims/scenery/issues/464 and
    // http://news.qooxdoo.org/mouse-capturing.
    const target = this.attachToWindow ? document.body : this.display.domElement;

    // NOTE: pointerId can be zero. It usually is zero for the mouse for Firefox.
    if ( target.setPointerCapture && !context.allowsDOMInput() ) {
      // NOTE: This will error out if run on a playback destination, where a pointer with the given ID does not exist.
      target.setPointerCapture( context.domEvent.pointerId );
    }

    type = this.handleUnknownPointerType( type, id );
    switch( type ) {
      case 'mouse':
        // The actual event afterwards
        this.mouseDown( id, point, context );
        break;
      case 'touch':
        this.touchStart( id, point, context );
        break;
      case 'pen':
        this.penStart( id, point, context );
        break;
      default:
        if ( assert ) {
          throw new Error( `Unknown pointer type: ${type}` );
        }
    }
  }

  /**
   * Handles a pointerup event, forwarding it to the proper logical event. (scenery-internal)
   */
  public pointerUp( id: number, type: string, point: Vector2, context: EventContext<PointerEvent> ): void {

    // update this outside of the Action executions so that PhET-iO event playback does not override it
    this.upTimeStamp = context.domEvent.timeStamp;

    type = this.handleUnknownPointerType( type, id );
    switch( type ) {
      case 'mouse':
        this.mouseUp( point, context );
        break;
      case 'touch':
        this.touchEnd( id, point, context );
        break;
      case 'pen':
        this.penEnd( id, point, context );
        break;
      default:
        if ( assert ) {
          throw new Error( `Unknown pointer type: ${type}` );
        }
    }
  }

  /**
   * Handles a pointercancel event, forwarding it to the proper logical event. (scenery-internal)
   */
  public pointerCancel( id: number, type: string, point: Vector2, context: EventContext<PointerEvent> ): void {
    type = this.handleUnknownPointerType( type, id );
    switch( type ) {
      case 'mouse':
        if ( console && console.log ) {
          console.log( 'WARNING: Pointer mouse cancel was received' );
        }
        break;
      case 'touch':
        this.touchCancel( id, point, context );
        break;
      case 'pen':
        this.penCancel( id, point, context );
        break;
      default:
        if ( console.log ) {
          console.log( `Unknown pointer type: ${type}` );
        }
    }
  }

  /**
   * Handles a gotpointercapture event, forwarding it to the proper logical event. (scenery-internal)
   */
  public gotPointerCapture( id: number, type: string, point: Vector2, context: EventContext ): void {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `gotPointerCapture('${id}',${Input.debugText( null, context.domEvent )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.gotPointerCaptureAction.execute( id, context );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Handles a lostpointercapture event, forwarding it to the proper logical event. (scenery-internal)
   */
  public lostPointerCapture( id: number, type: string, point: Vector2, context: EventContext ): void {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `lostPointerCapture('${id}',${Input.debugText( null, context.domEvent )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.lostPointerCaptureAction.execute( id, context );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Handles a pointermove event, forwarding it to the proper logical event. (scenery-internal)
   */
  public pointerMove( id: number, type: string, point: Vector2, context: EventContext<PointerEvent> ): void {
    type = this.handleUnknownPointerType( type, id );
    switch( type ) {
      case 'mouse':
        this.mouseMove( point, context );
        break;
      case 'touch':
        this.touchMove( id, point, context );
        break;
      case 'pen':
        this.penMove( id, point, context );
        break;
      default:
        if ( console.log ) {
          console.log( `Unknown pointer type: ${type}` );
        }
    }
  }

  /**
   * Handles a pointerover event, forwarding it to the proper logical event. (scenery-internal)
   */
  public pointerOver( id: number, type: string, point: Vector2, context: EventContext<PointerEvent> ): void {
    // TODO: accumulate mouse/touch info in the object if needed? https://github.com/phetsims/scenery/issues/1581
    // TODO: do we want to branch change on these types of events? https://github.com/phetsims/scenery/issues/1581
  }

  /**
   * Handles a pointerout event, forwarding it to the proper logical event. (scenery-internal)
   */
  public pointerOut( id: number, type: string, point: Vector2, context: EventContext<PointerEvent> ): void {
    // TODO: accumulate mouse/touch info in the object if needed? https://github.com/phetsims/scenery/issues/1581
    // TODO: do we want to branch change on these types of events? https://github.com/phetsims/scenery/issues/1581
  }

  /**
   * Handles a pointerenter event, forwarding it to the proper logical event. (scenery-internal)
   */
  public pointerEnter( id: number, type: string, point: Vector2, context: EventContext<PointerEvent> ): void {
    // TODO: accumulate mouse/touch info in the object if needed? https://github.com/phetsims/scenery/issues/1581
    // TODO: do we want to branch change on these types of events? https://github.com/phetsims/scenery/issues/1581
  }

  /**
   * Handles a pointerleave event, forwarding it to the proper logical event. (scenery-internal)
   */
  public pointerLeave( id: number, type: string, point: Vector2, context: EventContext<PointerEvent> ): void {
    // TODO: accumulate mouse/touch info in the object if needed? https://github.com/phetsims/scenery/issues/1581
    // TODO: do we want to branch change on these types of events? https://github.com/phetsims/scenery/issues/1581
  }

  /**
   * Handles a focusin event, forwarding it to the proper logical event. (scenery-internal)
   */
  public focusIn( context: EventContext<FocusEvent> ): void {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `focusIn('${Input.debugText( null, context.domEvent )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();

    this.focusinAction.execute( context );

    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Handles a focusout event, forwarding it to the proper logical event. (scenery-internal)
   */
  public focusOut( context: EventContext<FocusEvent> ): void {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `focusOut('${Input.debugText( null, context.domEvent )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();

    this.focusoutAction.execute( context );

    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Handles an input event, forwarding it to the proper logical event. (scenery-internal)
   */
  public input( context: EventContext<Event | InputEvent> ): void {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `input('${Input.debugText( null, context.domEvent )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();

    this.inputAction.execute( context );

    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Handles a change event, forwarding it to the proper logical event. (scenery-internal)
   */
  public change( context: EventContext ): void {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `change('${Input.debugText( null, context.domEvent )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();

    this.changeAction.execute( context );

    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Handles a click event, forwarding it to the proper logical event. (scenery-internal)
   */
  public click( context: EventContext<MouseEvent> ): void {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `click('${Input.debugText( null, context.domEvent )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();

    this.clickAction.execute( context );

    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Handles a keydown event, forwarding it to the proper logical event. (scenery-internal)
   */
  public keyDown( context: EventContext<KeyboardEvent> ): void {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `keyDown('${Input.debugText( null, context.domEvent )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();

    this.keydownAction.execute( context );

    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Handles a keyup event, forwarding it to the proper logical event. (scenery-internal)
   */
  public keyUp( context: EventContext<KeyboardEvent> ): void {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `keyUp('${Input.debugText( null, context.domEvent )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();

    this.keyupAction.execute( context );

    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * When we get an unknown pointer event type (allowed in the spec, see
   * https://developer.mozilla.org/en-US/docs/Web/API/PointerEvent/pointerType), we'll try to guess the pointer type
   * so that we can properly start/end the interaction. NOTE: this can happen for an 'up' where we received a
   * proper type for a 'down', so thus we need the detection.
   */
  private handleUnknownPointerType( type: string, id: number ): string {
    if ( type !== '' ) {
      return type;
    }
    return ( this.mouse && this.mouse.id === id ) ? 'mouse' : 'touch';
  }

  /**
   * Given a pointer reference, hit test it and determine the Trail that the pointer is over.
   */
  private getPointerTrail( pointer: Pointer ): Trail {
    return this.rootNode.trailUnderPointer( pointer ) || new Trail( this.rootNode );
  }

  /**
   * Called for each logical "up" event, for any pointer type.
   */
  private upEvent<DOMEvent extends Event>( pointer: Pointer, context: EventContext<DOMEvent>, point: Vector2 ): void {
    // if the event target is within the PDOM the AT is sending a fake pointer event to the document - do not
    // dispatch this since the PDOM should only handle Input.PDOM_EVENT_TYPES, and all other pointer input should
    // go through the Display div. Otherwise, activation will be duplicated when we handle pointer and PDOM events
    if ( this.display.isElementUnderPDOM( context.domEvent.target as HTMLElement, true ) ) {
      return;
    }

    const pointChanged = pointer.up( point, context.domEvent );

    sceneryLog && sceneryLog.Input && sceneryLog.Input( `upEvent ${pointer.toString()} changed:${pointChanged}` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();

    // We'll use this trail for the entire dispatch of this event.
    const eventTrail = this.branchChangeEvents<DOMEvent>( pointer, context, pointChanged );

    this.dispatchEvent<DOMEvent>( eventTrail, 'up', pointer, context, true );

    // touch pointers are transient, so fire exit/out to the trail afterwards
    if ( pointer.isTouchLike() ) {
      this.exitEvents<DOMEvent>( pointer, context, eventTrail, 0, true );
    }

    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Called for each logical "down" event, for any pointer type.
   */
  private downEvent<DOMEvent extends Event>( pointer: Pointer, context: EventContext<DOMEvent>, point: Vector2 ): void {
    // if the event target is within the PDOM the AT is sending a fake pointer event to the document - do not
    // dispatch this since the PDOM should only handle Input.PDOM_EVENT_TYPES, and all other pointer input should
    // go through the Display div. Otherwise, activation will be duplicated when we handle pointer and PDOM events
    if ( this.display.isElementUnderPDOM( context.domEvent.target as HTMLElement, true ) ) {
      return;
    }

    const pointChanged = pointer.updatePoint( point );

    sceneryLog && sceneryLog.Input && sceneryLog.Input( `downEvent ${pointer.toString()} changed:${pointChanged}` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();

    // We'll use this trail for the entire dispatch of this event.
    const eventTrail = this.branchChangeEvents<DOMEvent>( pointer, context, pointChanged );

    pointer.down( context.domEvent );

    this.dispatchEvent<DOMEvent>( eventTrail, 'down', pointer, context, true );

    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Called for each logical "move" event, for any pointer type.
   */
  private moveEvent<DOMEvent extends Event>( pointer: Pointer, context: EventContext<DOMEvent> ): void {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `moveEvent ${pointer.toString()}` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();

    // Always treat move events as "point changed"
    this.branchChangeEvents<DOMEvent>( pointer, context, true );

    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Called for each logical "cancel" event, for any pointer type.
   */
  private cancelEvent<DOMEvent extends Event>( pointer: Pointer, context: EventContext<DOMEvent>, point: Vector2 ): void {
    const pointChanged = pointer.cancel( point );

    sceneryLog && sceneryLog.Input && sceneryLog.Input( `cancelEvent ${pointer.toString()} changed:${pointChanged}` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();

    // We'll use this trail for the entire dispatch of this event.
    const eventTrail = this.branchChangeEvents<DOMEvent>( pointer, context, pointChanged );

    this.dispatchEvent<DOMEvent>( eventTrail, 'cancel', pointer, context, true );

    // touch pointers are transient, so fire exit/out to the trail afterwards
    if ( pointer.isTouchLike() ) {
      this.exitEvents<DOMEvent>( pointer, context, eventTrail, 0, true );
    }

    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Dispatches any necessary events that would result from the pointer's trail changing.
   *
   * This will send the necessary exit/enter events (on subtrails that have diverged between before/after), the
   * out/over events, and if flagged a move event.
   *
   * @param pointer
   * @param context
   * @param sendMove - Whether to send move events
   * @returns - The current trail of the pointer
   */
  private branchChangeEvents<DOMEvent extends Event>( pointer: Pointer, context: EventContext<DOMEvent>, sendMove: boolean ): Trail {
    sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent(
      `branchChangeEvents: ${pointer.toString()} sendMove:${sendMove}` );
    sceneryLog && sceneryLog.InputEvent && sceneryLog.push();

    const trail = this.getPointerTrail( pointer );

    const inputEnabledTrail = trail.slice( 0, Math.min( trail.nodes.length, trail.getLastInputEnabledIndex() + 1 ) );
    const oldInputEnabledTrail = pointer.inputEnabledTrail || new Trail( this.rootNode );
    const branchInputEnabledIndex = Trail.branchIndex( inputEnabledTrail, oldInputEnabledTrail );
    const lastInputEnabledNodeChanged = oldInputEnabledTrail.lastNode() !== inputEnabledTrail.lastNode();

    if ( sceneryLog && sceneryLog.InputEvent ) {
      const oldTrail = pointer.trail || new Trail( this.rootNode );
      const branchIndex = Trail.branchIndex( trail, oldTrail );

      ( branchIndex !== trail.length || branchIndex !== oldTrail.length ) && sceneryLog.InputEvent(
        `changed from ${oldTrail.toString()} to ${trail.toString()}` );
    }

    // event order matches http://www.w3.org/TR/DOM-Level-3-Events/#events-mouseevent-event-order
    if ( sendMove ) {
      this.dispatchEvent<DOMEvent>( trail, 'move', pointer, context, true );
    }

    // We want to approximately mimic http://www.w3.org/TR/DOM-Level-3-Events/#events-mouseevent-event-order
    this.exitEvents<DOMEvent>( pointer, context, oldInputEnabledTrail, branchInputEnabledIndex, lastInputEnabledNodeChanged );
    this.enterEvents<DOMEvent>( pointer, context, inputEnabledTrail, branchInputEnabledIndex, lastInputEnabledNodeChanged );

    pointer.trail = trail;
    pointer.inputEnabledTrail = inputEnabledTrail;

    sceneryLog && sceneryLog.InputEvent && sceneryLog.pop();
    return trail;
  }

  /**
   * Triggers 'enter' events along a trail change, and an 'over' event on the leaf.
   *
   * For example, if we change from a trail [ a, b, c, d, e ] => [ a, b, x, y ], it will fire:
   *
   * - enter x
   * - enter y
   * - over y (bubbles)
   *
   * @param pointer
   * @param event
   * @param trail - The "new" trail
   * @param branchIndex - The first index where the old and new trails have a different node. We will notify
   *                               for this node and all "descendant" nodes in the relevant trail.
   * @param lastNodeChanged - If the last node didn't change, we won't sent an over event.
   */
  private enterEvents<DOMEvent extends Event>( pointer: Pointer, context: EventContext<DOMEvent>, trail: Trail, branchIndex: number, lastNodeChanged: boolean ): void {
    if ( lastNodeChanged ) {
      this.dispatchEvent<DOMEvent>( trail, 'over', pointer, context, true, true );
    }

    for ( let i = branchIndex; i < trail.length; i++ ) {
      this.dispatchEvent<DOMEvent>( trail.slice( 0, i + 1 ), 'enter', pointer, context, false );
    }
  }

  /**
   * Triggers 'exit' events along a trail change, and an 'out' event on the leaf.
   *
   * For example, if we change from a trail [ a, b, c, d, e ] => [ a, b, x, y ], it will fire:
   *
   * - out e (bubbles)
   * - exit c
   * - exit d
   * - exit e
   *
   * @param pointer
   * @param event
   * @param trail - The "old" trail
   * @param branchIndex - The first index where the old and new trails have a different node. We will notify
   *                               for this node and all "descendant" nodes in the relevant trail.
   * @param lastNodeChanged - If the last node didn't change, we won't sent an out event.
   */
  private exitEvents<DOMEvent extends Event>( pointer: Pointer, context: EventContext<DOMEvent>, trail: Trail, branchIndex: number, lastNodeChanged: boolean ): void {
    for ( let i = trail.length - 1; i >= branchIndex; i-- ) {
      this.dispatchEvent<DOMEvent>( trail.slice( 0, i + 1 ), 'exit', pointer, context, false, true );
    }

    if ( lastNodeChanged ) {
      this.dispatchEvent<DOMEvent>( trail, 'out', pointer, context, true );
    }
  }

  /**
   * Dispatch to all nodes in the Trail, optionally bubbling down from the leaf to the root.
   *
   * @param trail
   * @param type
   * @param pointer
   * @param context
   * @param bubbles - If bubbles is false, the event is only dispatched to the leaf node of the trail.
   * @param fireOnInputDisabled - Whether to fire this event even if nodes have inputEnabled:false
   * @param dispatchToTargets - Whether to fire the event on targets along the Trail. If false, only Display and Pointer listeners are fired.
   */
  private dispatchEvent<DOMEvent extends Event>(
    trail: Trail,
    type: SupportedEventTypes,
    pointer: Pointer,
    context: EventContext<DOMEvent>,
    bubbles: boolean,
    fireOnInputDisabled = false,
    dispatchToTargets = true ): void {
    sceneryLog && sceneryLog.EventDispatch && sceneryLog.EventDispatch(
      `${type} trail:${trail.toString()} pointer:${pointer.toString()} at ${pointer.point ? pointer.point.toString() : 'null'}` );
    sceneryLog && sceneryLog.EventDispatch && sceneryLog.push();

    assert && assert( trail, 'Falsy trail for dispatchEvent' );

    sceneryLog && sceneryLog.EventPath && sceneryLog.EventPath( `${type} ${trail.toPathString()}` );

    // NOTE: event is not immutable, as its currentTarget changes
    const inputEvent = new SceneryEvent<DOMEvent>( trail, type, pointer, context );

    this.currentSceneryEvent = inputEvent;

    // first run through the pointer's listeners to see if one of them will handle the event
    this.dispatchToListeners<DOMEvent>( pointer, pointer.getListeners(), type, inputEvent );

    // if not yet handled, run through the trail in order to see if one of them will handle the event
    // at the base of the trail should be the scene node, so the scene will be notified last
    dispatchToTargets && this.dispatchToTargets<DOMEvent>( trail, type, pointer, inputEvent, bubbles, fireOnInputDisabled );

    // Notify input listeners on the Display
    this.dispatchToListeners<DOMEvent>( pointer, this.display.getInputListeners(), type, inputEvent );

    // Notify input listeners to any Display
    if ( DisplayGlobals.inputListeners.length ) {
      this.dispatchToListeners<DOMEvent>( pointer, DisplayGlobals.inputListeners.slice(), type, inputEvent );
    }

    this.currentSceneryEvent = null;

    sceneryLog && sceneryLog.EventDispatch && sceneryLog.pop();
  }

  /**
   * Notifies an array of listeners with a specific event.
   *
   * @param pointer
   * @param listeners - Should be a defensive array copy already.
   * @param type
   * @param inputEvent
   */
  private dispatchToListeners<DOMEvent extends Event>( pointer: Pointer, listeners: TInputListener[], type: SupportedEventTypes, inputEvent: SceneryEvent<DOMEvent> ): void {

    if ( inputEvent.handled ) {
      return;
    }

    const specificType = pointer.type + type as SupportedEventTypes; // e.g. mouseup, touchup

    for ( let i = 0; i < listeners.length; i++ ) {
      const listener = listeners[ i ];

      if ( !inputEvent.aborted && listener[ specificType as keyof TInputListener ] ) {
        sceneryLog && sceneryLog.EventDispatch && sceneryLog.EventDispatch( specificType );
        sceneryLog && sceneryLog.EventDispatch && sceneryLog.push();

        ( listener[ specificType as keyof TInputListener ] as SceneryListenerFunction<DOMEvent> )( inputEvent );

        sceneryLog && sceneryLog.EventDispatch && sceneryLog.pop();
      }

      if ( !inputEvent.aborted && listener[ type as keyof TInputListener ] ) {
        sceneryLog && sceneryLog.EventDispatch && sceneryLog.EventDispatch( type );
        sceneryLog && sceneryLog.EventDispatch && sceneryLog.push();

        ( listener[ type as keyof TInputListener ] as SceneryListenerFunction<DOMEvent> )( inputEvent );

        sceneryLog && sceneryLog.EventDispatch && sceneryLog.pop();
      }
    }
  }

  /**
   * Dispatch to all nodes in the Trail, optionally bubbling down from the leaf to the root.
   *
   * @param trail
   * @param type
   * @param pointer
   * @param inputEvent
   * @param bubbles - If bubbles is false, the event is only dispatched to the leaf node of the trail.
   * @param [fireOnInputDisabled]
   */
  private dispatchToTargets<DOMEvent extends Event>( trail: Trail, type: SupportedEventTypes, pointer: Pointer,
                                                     inputEvent: SceneryEvent<DOMEvent>, bubbles: boolean,
                                                     fireOnInputDisabled = false ): void {

    if ( inputEvent.aborted || inputEvent.handled ) {
      return;
    }

    const inputEnabledIndex = trail.getLastInputEnabledIndex();

    for ( let i = trail.nodes.length - 1; i >= 0; bubbles ? i-- : i = -1 ) {

      const target = trail.nodes[ i ];

      const trailInputDisabled = inputEnabledIndex < i;

      if ( target.isDisposed || ( !fireOnInputDisabled && trailInputDisabled ) ) {
        continue;
      }

      inputEvent.currentTarget = target;

      this.dispatchToListeners<DOMEvent>( pointer, target.getInputListeners(), type, inputEvent );

      // if the input event was aborted or handled, don't follow the trail down another level
      if ( inputEvent.aborted || inputEvent.handled ) {
        return;
      }
    }
  }

  /**
   * Convenience function for logging out a point/event combination.
   *
   * @param point - Not logged if null
   * @param domEvent
   */
  private static debugText( point: Vector2 | null, domEvent: Event ): string {
    let result = `${domEvent.timeStamp} ${domEvent.type}`;
    if ( point !== null ) {
      result = `${point.x},${point.y} ${result}`;
    }
    return result;
  }
}

scenery.register( 'Input', Input );