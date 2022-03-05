// Copyright 2013-2022, University of Colorado Boulder

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

import Action from '../../../axon/js/Action.js';
import TinyEmitter from '../../../axon/js/TinyEmitter.js';
import Vector2 from '../../../dot/js/Vector2.js';
import cleanArray from '../../../phet-core/js/cleanArray.js';
import optionize from '../../../phet-core/js/optionize.js';
import platform from '../../../phet-core/js/platform.js';
import EventType from '../../../tandem/js/EventType.js';
import Tandem from '../../../tandem/js/Tandem.js';
import NullableIO from '../../../tandem/js/types/NullableIO.js';
import NumberIO from '../../../tandem/js/types/NumberIO.js';
import { scenery, Display, Features, Trail, EventIO, Mouse, PDOMPointer, Pen, Pointer, SceneryEvent, Touch, PDOMUtils, BatchedDOMEvent, BrowserEvents, Node, BatchedDOMEventType, BatchedDOMEventCallback, IInputListener, SceneryListenerFunction, WindowTouch } from '../imports.js';

// Object literal makes it easy to check for the existence of an attribute (compared to [].indexOf()>=0). Helpful for
// serialization. NOTE: Do not add or change this without consulting the PhET-iO IOType schema for this in EventIO.js

const domEventPropertiesToSerialize = [
  'type',
  'button', 'keyCode', 'key',
  'deltaX', 'deltaY', 'deltaZ', 'deltaMode', 'pointerId',
  'pointerType', 'charCode', 'which', 'clientX', 'clientY', 'pageX', 'pageY', 'changedTouches',
  'scale',
  'target', 'relatedTarget',
  'ctrlKey', 'shiftKey', 'altKey', 'metaKey', 'code'
];

// A list of keys on events that need to be serialized into HTMLElements
const EVENT_KEY_VALUES_AS_ELEMENTS = [ 'target', 'relatedTarget' ];

// A list of events that should still fire, even when the Node is not pickable
const PDOM_UNPICKABLE_EVENTS = [ 'focus', 'blur', 'focusin', 'focusout' ];
const TARGET_SUBSTITUTE_KEY = 'targetSubstitute' as const;
const RELATED_TARGET_SUBSTITUTE_KEY = 'relatedTargetSubstitute' as const;

// A bit more than the maximum amount of time that iOS 14 VoiceOver was observed to delay between
// sending a mouseup event and a click event.
const PDOM_CLICK_DELAY = 80;

type InputOptions = {
  tandem?: Tandem;
};

type EventListenerOptions = { capture?: boolean, passive?: boolean, once?: boolean } | boolean;

class Input {

  display: Display;
  rootNode: Node;

  attachToWindow: boolean;
  batchDOMEvents: boolean;
  assumeFullWindow: boolean;
  passiveEvents: boolean | null;

  // Pointer for accessibility, only created lazily on first pdom event.
  pdomPointer: PDOMPointer | null;

  // Pointer for mouse, only created lazily on first mouse event, so no mouse is allocated on tablets.
  mouse: Mouse | null;

  // All active pointers.
  pointers: Pointer[];

  pointerAddedEmitter: TinyEmitter<[ Pointer ]>;

  // Whether we are currently firing events. We need to track this to handle re-entrant cases
  // like https://github.com/phetsims/balloons-and-static-electricity/issues/406.
  currentlyFiringEvents: boolean;

  private batchedEvents: BatchedDOMEvent[];

  // In miliseconds, the DOMEvent timeStamp when we receive a logical up event.
  // We can compare this to the timeStamp on a click vent to filter out the click events
  // when some screen readers send both down/up events AND click events to the target
  // element, see https://github.com/phetsims/scenery/issues/1094
  private upTimeStamp: number;

  // Emits pointer validation to the input stream for playback
  // This is a high frequency event that is necessary for reproducible playbacks
  private validatePointersAction: Action;

  private mouseUpAction: Action<[ Vector2, MouseEvent ]>;
  private mouseDownAction: Action<[ number, Vector2, MouseEvent ]>;
  private mouseMoveAction: Action<[ Vector2, MouseEvent ]>;
  private mouseOverAction: Action<[ Vector2, MouseEvent ]>;
  private mouseOutAction: Action<[ Vector2, MouseEvent ]>;
  private wheelScrollAction: Action<[ WheelEvent ]>;
  private touchStartAction: Action<[ number, Vector2, TouchEvent | PointerEvent ]>;
  private touchEndAction: Action<[ number, Vector2, TouchEvent | PointerEvent ]>;
  private touchMoveAction: Action<[ number, Vector2, TouchEvent | PointerEvent ]>;
  private touchCancelAction: Action<[ number, Vector2, TouchEvent | PointerEvent ]>;
  private penStartAction: Action<[ number, Vector2, PointerEvent ]>;
  private penEndAction: Action<[ number, Vector2, PointerEvent ]>;
  private penMoveAction: Action<[ number, Vector2, PointerEvent ]>;
  private penCancelAction: Action<[ number, Vector2, PointerEvent ]>;
  private gotPointerCaptureAction: Action<[ number, Event ]>;
  private lostPointerCaptureAction: Action<[ number, Event ]>;

  // If accessible
  private focusinAction?: Action<[ FocusEvent ]>;
  private focusoutAction?: Action<[ FocusEvent ]>;
  private clickAction?: Action<[ MouseEvent ]>;
  private inputAction?: Action<[ Event | InputEvent ]>;
  private changeAction?: Action<[ Event ]>;
  private keydownAction?: Action<[ KeyboardEvent ]>;
  private keyupAction?: Action<[ KeyboardEvent ]>;

  // Same event options for all DOM listeners, used when we connect listeners
  private accessibleEventOptions?: EventListenerOptions;

  // Maps events that are added to the root PDOM element to the listener that will
  // fire one of the above Actions and finally dispatch a corresponding SceneryEvent to scenery targets.
  // Event listeners are not added until initializeEvents, and are stored in this Map so they can be removed
  // again in detachEvents.
  private pdomEventListenerMap?: Map<string, ( event: Event ) => void>;


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
  constructor( display: Display, attachToWindow: boolean, batchDOMEvents: boolean, assumeFullWindow: boolean, passiveEvents: boolean | null, providedOptions?: InputOptions ) {
    assert && assert( display instanceof Display );
    assert && assert( typeof attachToWindow === 'boolean' );
    assert && assert( typeof batchDOMEvents === 'boolean' );
    assert && assert( typeof assumeFullWindow === 'boolean' );

    const options = optionize<InputOptions, InputOptions>( {
      tandem: Tandem.OPTIONAL
    }, providedOptions );

    this.display = display;
    this.rootNode = display.rootNode;

    this.attachToWindow = attachToWindow;
    this.batchDOMEvents = batchDOMEvents;
    this.assumeFullWindow = assumeFullWindow;
    this.passiveEvents = passiveEvents;
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

    this.validatePointersAction = new Action( () => {
      let i = this.pointers.length;
      while ( i-- ) {
        const pointer = this.pointers[ i ];
        if ( pointer.point && pointer !== this.pdomPointer ) {
          this.branchChangeEvents<Event>( pointer, pointer.lastDOMEvent, false );
        }
      }
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'validatePointersAction' ),
      phetioHighFrequency: true
    } );

    this.mouseUpAction = new Action( ( point: Vector2, event: MouseEvent ) => {
      const mouse = this.ensureMouse( point );
      const pointChanged = mouse.up( point, event );
      mouse.id = null;
      this.upEvent<MouseEvent>( mouse, event, pointChanged );
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'mouseUpAction' ),
      parameters: [
        { name: 'point', phetioType: Vector2.Vector2IO },
        { name: 'event', phetioType: EventIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when a mouse button is released.'
    } );

    this.mouseDownAction = new Action( ( id: number, point: Vector2, event: MouseEvent ) => {
      const mouse = this.ensureMouse( point );
      mouse.id = id;
      const pointChanged = mouse.down( point, event );
      this.downEvent<MouseEvent>( mouse, event, pointChanged );
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'mouseDownAction' ),
      parameters: [
        { name: 'id', phetioType: NullableIO( NumberIO ) },
        { name: 'point', phetioType: Vector2.Vector2IO },
        { name: 'event', phetioType: EventIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when a mouse button is pressed.'
    } );

    this.mouseMoveAction = new Action( ( point: Vector2, event: MouseEvent ) => {
      const mouse = this.ensureMouse( point );
      mouse.move( point, event );
      this.moveEvent<MouseEvent>( mouse, event );
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'mouseMoveAction' ),
      parameters: [
        { name: 'point', phetioType: Vector2.Vector2IO },
        { name: 'event', phetioType: EventIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when the mouse is moved.',
      phetioHighFrequency: true
    } );

    this.mouseOverAction = new Action( ( point: Vector2, event: MouseEvent ) => {
      const mouse = this.ensureMouse( point );
      mouse.over( point, event );
      // TODO: how to handle mouse-over (and log it)... are we changing the pointer.point without a branch change?
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'mouseOverAction' ),
      parameters: [
        { name: 'point', phetioType: Vector2.Vector2IO },
        { name: 'event', phetioType: EventIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when the mouse is moved while on the sim.'
    } );

    this.mouseOutAction = new Action( ( point: Vector2, event: MouseEvent ) => {
      const mouse = this.ensureMouse( point );
      mouse.out( point, event );
      // TODO: how to handle mouse-out (and log it)... are we changing the pointer.point without a branch change?
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'mouseOutAction' ),
      parameters: [
        { name: 'point', phetioType: Vector2.Vector2IO },
        { name: 'event', phetioType: EventIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when the mouse moves out of the display.'
    } );

    this.wheelScrollAction = new Action( ( event: WheelEvent ) => {
      const mouse = this.ensureMouse( this.pointFromEvent( event ) );
      mouse.wheel( event );

      // don't send mouse-wheel events if we don't yet have a mouse location!
      // TODO: Can we set the mouse location based on the wheel event?
      if ( mouse.point ) {
        const trail = this.rootNode.trailUnderPointer( mouse ) || new Trail( this.rootNode );
        this.dispatchEvent<WheelEvent>( trail, 'wheel', mouse, event, true );
      }
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'wheelScrollAction' ),
      parameters: [
        { name: 'event', phetioType: EventIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when the mouse wheel scrolls.',
      phetioHighFrequency: true
    } );

    this.touchStartAction = new Action( ( id: number, point: Vector2, event: TouchEvent | PointerEvent ) => {
      const touch = new Touch( id, point, event );
      this.addPointer( touch );
      this.downEvent<TouchEvent | PointerEvent>( touch, event, false );
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'touchStartAction' ),
      parameters: [
        { name: 'id', phetioType: NumberIO },
        { name: 'point', phetioType: Vector2.Vector2IO },
        { name: 'event', phetioType: EventIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when a touch begins.'
    } );

    this.touchEndAction = new Action( ( id: number, point: Vector2, event: TouchEvent | PointerEvent ) => {
      const touch = this.findPointerById( id ) as Touch | null;
      if ( touch ) {
        assert && assert( touch instanceof Touch ); // eslint-disable-line
        const pointChanged = touch.end( point, event );
        this.upEvent<TouchEvent | PointerEvent>( touch, event, pointChanged );
        this.removePointer( touch );
      }
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'touchEndAction' ),
      parameters: [
        { name: 'id', phetioType: NumberIO },
        { name: 'point', phetioType: Vector2.Vector2IO },
        { name: 'event', phetioType: EventIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when a touch ends.'
    } );

    this.touchMoveAction = new Action( ( id: number, point: Vector2, event: TouchEvent | PointerEvent ) => {
      const touch = this.findPointerById( id ) as Touch | null;
      if ( touch ) {
        assert && assert( touch instanceof Touch ); // eslint-disable-line
        touch.move( point, event );
        this.moveEvent<TouchEvent | PointerEvent>( touch, event );
      }
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'touchMoveAction' ),
      parameters: [
        { name: 'id', phetioType: NumberIO },
        { name: 'point', phetioType: Vector2.Vector2IO },
        { name: 'event', phetioType: EventIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when a touch moves.',
      phetioHighFrequency: true
    } );

    this.touchCancelAction = new Action( ( id: number, point: Vector2, event: TouchEvent | PointerEvent ) => {
      const touch = this.findPointerById( id ) as Touch | null;
      if ( touch ) {
        assert && assert( touch instanceof Touch ); // eslint-disable-line
        const pointChanged = touch.cancel( point, event );
        this.cancelEvent<TouchEvent | PointerEvent>( touch, event, pointChanged );
        this.removePointer( touch );
      }
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'touchCancelAction' ),
      parameters: [
        { name: 'id', phetioType: NumberIO },
        { name: 'point', phetioType: Vector2.Vector2IO },
        { name: 'event', phetioType: EventIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when a touch is canceled.'
    } );

    this.penStartAction = new Action( ( id: number, point: Vector2, event: PointerEvent ) => {
      const pen = new Pen( id, point, event );
      this.addPointer( pen );
      this.downEvent<PointerEvent>( pen, event, false );
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'penStartAction' ),
      parameters: [
        { name: 'id', phetioType: NumberIO },
        { name: 'point', phetioType: Vector2.Vector2IO },
        { name: 'event', phetioType: EventIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when a pen touches the screen.'
    } );

    this.penEndAction = new Action( ( id: number, point: Vector2, event: PointerEvent ) => {
      const pen = this.findPointerById( id ) as Pen | null;
      if ( pen ) {
        assert && assert( pen instanceof Pen );
        const pointChanged = pen.end( point, event );
        this.upEvent<PointerEvent>( pen, event, pointChanged );
        this.removePointer( pen );
      }
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'penEndAction' ),
      parameters: [
        { name: 'id', phetioType: NumberIO },
        { name: 'point', phetioType: Vector2.Vector2IO },
        { name: 'event', phetioType: EventIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when a pen is lifted.'
    } );

    this.penMoveAction = new Action( ( id: number, point: Vector2, event: PointerEvent ) => {
      const pen = this.findPointerById( id ) as Pen | null;
      if ( pen ) {
        assert && assert( pen instanceof Pen );
        pen.move( point, event );
        this.moveEvent<PointerEvent>( pen, event );
      }
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'penMoveAction' ),
      parameters: [
        { name: 'id', phetioType: NumberIO },
        { name: 'point', phetioType: Vector2.Vector2IO },
        { name: 'event', phetioType: EventIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when a pen is moved.',
      phetioHighFrequency: true
    } );

    this.penCancelAction = new Action( ( id: number, point: Vector2, event: PointerEvent ) => {
      const pen = this.findPointerById( id ) as Pen | null;
      if ( pen ) {
        assert && assert( pen instanceof Pen );
        const pointChanged = pen.cancel( point, event );
        this.cancelEvent<PointerEvent>( pen, event, pointChanged );
        this.removePointer( pen );
      }
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'penCancelAction' ),
      parameters: [
        { name: 'id', phetioType: NumberIO },
        { name: 'point', phetioType: Vector2.Vector2IO },
        { name: 'event', phetioType: EventIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when a pen is canceled.'
    } );

    this.gotPointerCaptureAction = new Action( ( id: number, event: Event ) => {
      const pointer = this.findPointerById( id );

      if ( pointer ) {
        pointer.onGotPointerCapture();
      }
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'gotPointerCaptureAction' ),
      parameters: [
        { name: 'id', phetioType: NumberIO },
        { name: 'event', phetioType: EventIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when a pointer is captured (normally at the start of an interaction)',
      phetioHighFrequency: true
    } );

    this.lostPointerCaptureAction = new Action( ( id: number, event: Event ) => {
      const pointer = this.findPointerById( id );

      if ( pointer ) {
        pointer.onLostPointerCapture();
      }
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'lostPointerCaptureAction' ),
      parameters: [
        { name: 'id', phetioType: NumberIO },
        { name: 'event', phetioType: EventIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when a pointer loses its capture (normally at the end of an interaction)',
      phetioHighFrequency: true
    } );

    // wire up accessibility listeners on the display's root accessible DOM element.
    if ( this.display._accessible ) {
      this.focusinAction = new Action( ( event: FocusEvent ) => {

        // ignore any focusout callbacks if they are initiated due to implementation details in PDOM manipulation
        if ( this.display.blockFocusCallbacks ) {
          return;
        }

        sceneryLog && sceneryLog.Input && sceneryLog.Input( `focusin(${Input.debugText( null, event )});` );
        sceneryLog && sceneryLog.Input && sceneryLog.push();

        const trail = this.updateTrailForPDOMDispatch( event );
        this.dispatchPDOMEvent<FocusEvent>( trail, 'focus', event, false );
        this.dispatchPDOMEvent<FocusEvent>( trail, 'focusin', event, true );

        sceneryLog && sceneryLog.Input && sceneryLog.pop();
      }, {
        phetioPlayback: true,
        tandem: options.tandem.createTandem( 'focusinAction' ),
        parameters: [
          { name: 'event', phetioType: EventIO }
        ],
        phetioEventType: EventType.USER,
        phetioDocumentation: 'Emits when the PDOM root gets the focusin DOM event.'
      } );

      this.focusoutAction = new Action( ( event: FocusEvent ) => {

        // ignore any focusout callbacks if they are initiated due to implementation details in PDOM manipulation
        if ( this.display.blockFocusCallbacks ) {
          return;
        }

        sceneryLog && sceneryLog.Input && sceneryLog.Input( `focusOut(${Input.debugText( null, event )});` );
        sceneryLog && sceneryLog.Input && sceneryLog.push();

        // recompute the trail on focusout if necessary - since a blur/focusout may have been initiated from a
        // focus/focusin listener, it is possible that focusout was called more than once before focusin is called on the
        // next active element, see https://github.com/phetsims/scenery/issues/898
        const pdomPointer = this.ensurePDOMPointer();
        pdomPointer.invalidateTrail( this.getTrailId( event ) );

        const trail = this.updateTrailForPDOMDispatch( event );
        this.dispatchPDOMEvent<FocusEvent>( trail, 'blur', event, false );
        this.dispatchPDOMEvent<FocusEvent>( trail, 'focusout', event, true );

        sceneryLog && sceneryLog.Input && sceneryLog.pop();
      }, {
        phetioPlayback: true,
        tandem: options.tandem.createTandem( 'focusoutAction' ),
        parameters: [
          { name: 'event', phetioType: EventIO }
        ],
        phetioEventType: EventType.USER,
        phetioDocumentation: 'Emits when the PDOM root gets the focusout DOM event.'
      } );

      // https://developer.mozilla.org/en-US/docs/Web/API/Element/click_event notes that the click action should result
      // in a MouseEvent
      this.clickAction = new Action( ( event: MouseEvent ) => {
        sceneryLog && sceneryLog.Input && sceneryLog.Input( `click(${Input.debugText( null, event )});` );
        sceneryLog && sceneryLog.Input && sceneryLog.push();

        const trail = this.updateTrailForPDOMDispatch( event );
        this.dispatchPDOMEvent<MouseEvent>( trail, 'click', event, true );

        sceneryLog && sceneryLog.Input && sceneryLog.pop();
      }, {
        phetioPlayback: true,
        tandem: options.tandem.createTandem( 'clickAction' ),
        parameters: [
          { name: 'event', phetioType: EventIO }
        ],
        phetioEventType: EventType.USER,
        phetioDocumentation: 'Emits when the PDOM root gets the click DOM event.'
      } );

      this.inputAction = new Action( ( event: Event | InputEvent ) => {
        sceneryLog && sceneryLog.Input && sceneryLog.Input( `input(${Input.debugText( null, event )});` );
        sceneryLog && sceneryLog.Input && sceneryLog.push();

        const trail = this.updateTrailForPDOMDispatch( event );
        this.dispatchPDOMEvent<Event | InputEvent>( trail, 'input', event, true );

        sceneryLog && sceneryLog.Input && sceneryLog.pop();
      }, {
        phetioPlayback: true,
        tandem: options.tandem.createTandem( 'inputAction' ),
        parameters: [
          { name: 'event', phetioType: EventIO }
        ],
        phetioEventType: EventType.USER,
        phetioDocumentation: 'Emits when the PDOM root gets the input DOM event.'
      } );

      this.changeAction = new Action( ( event: Event ) => {
        sceneryLog && sceneryLog.Input && sceneryLog.Input( `change(${Input.debugText( null, event )});` );
        sceneryLog && sceneryLog.Input && sceneryLog.push();

        const trail = this.updateTrailForPDOMDispatch( event );
        this.dispatchPDOMEvent<Event>( trail, 'change', event, true );

        sceneryLog && sceneryLog.Input && sceneryLog.pop();
      }, {
        phetioPlayback: true,
        tandem: options.tandem.createTandem( 'changeAction' ),
        parameters: [
          { name: 'event', phetioType: EventIO }
        ],
        phetioEventType: EventType.USER,
        phetioDocumentation: 'Emits when the PDOM root gets the change DOM event.'
      } );

      this.keydownAction = new Action( ( event: KeyboardEvent ) => {
        sceneryLog && sceneryLog.Input && sceneryLog.Input( `keydown(${Input.debugText( null, event )});` );
        sceneryLog && sceneryLog.Input && sceneryLog.push();

        const trail = this.updateTrailForPDOMDispatch( event );
        this.dispatchPDOMEvent<KeyboardEvent>( trail, 'keydown', event, true );

        sceneryLog && sceneryLog.Input && sceneryLog.pop();
      }, {
        phetioPlayback: true,
        tandem: options.tandem.createTandem( 'keydownAction' ),
        parameters: [
          { name: 'event', phetioType: EventIO }
        ],
        phetioEventType: EventType.USER,
        phetioDocumentation: 'Emits when the PDOM root gets the keydown DOM event.'
      } );

      this.keyupAction = new Action( ( event: KeyboardEvent ) => {
        sceneryLog && sceneryLog.Input && sceneryLog.Input( `keyup(${Input.debugText( null, event )});` );
        sceneryLog && sceneryLog.Input && sceneryLog.push();

        const trail = this.updateTrailForPDOMDispatch( event );
        this.dispatchPDOMEvent<KeyboardEvent>( trail, 'keyup', event, true );

        sceneryLog && sceneryLog.Input && sceneryLog.pop();
      }, {
        phetioPlayback: true,
        tandem: options.tandem.createTandem( 'keyupAction' ),
        parameters: [
          { name: 'event', phetioType: EventIO }
        ],
        phetioEventType: EventType.USER,
        phetioDocumentation: 'Emits when the PDOM root gets the keyup DOM event.'
      } );

      this.accessibleEventOptions = Features.passive ? { capture: false, passive: false } : false;
      this.pdomEventListenerMap = new Map();

      PDOMUtils.DOM_EVENTS.forEach( eventName => {

        const actionName = `${eventName}Action`;
        assert && assert( this[ actionName as keyof Input ], `action not defined on Input: ${actionName}` );

        this.pdomEventListenerMap!.set( eventName, event => {

          sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent( `Input.${eventName}FromBrowser` );
          sceneryLog && sceneryLog.InputEvent && sceneryLog.push();

          if ( this.display.interactive ) {

            const trailId = this.getTrailId( event );
            const trail = trailId ? Trail.fromUniqueId( this.display.rootNode, trailId ) : null;

            // Only dispatch the event if the click did not happen rapidly after an up event. It is
            // likely that the screen reader dispatched both pointer AND click events in this case, and
            // we only want to respond to one or the other. See https://github.com/phetsims/scenery/issues/1094.
            // This is outside of the clickAction execution so that blocked clicks are not part of the PhET-iO data
            // stream.
            if ( trail && !( _.some( trail.nodes, node => node.positionInPDOM ) && eventName === 'click' &&
                 event.timeStamp - this.upTimeStamp <= PDOM_CLICK_DELAY ) ) {
              ( this[ actionName as keyof Input ] as unknown as Action<[ Event ]> ).execute( event );
            }
          }

          sceneryLog && sceneryLog.InputEvent && sceneryLog.pop();
        } );
      } );
    }
  }

  /**
   * Interrupts any input actions that are currently taking place (should stop drags, etc.)
   */
  interruptPointers() {
    _.each( this.pointers, pointer => {
      pointer.interruptAll();
    } );
  }

  /**
   * Called to batch a raw DOM event (which may be immediately fired, depending on the settings). (scenery-internal)
   *
   * @param domEvent
   * @param batchType - See BatchedDOMEvent's "enumeration"
   * @param callback - Parameter types defined by the batchType. See BatchedDOMEvent for details
   * @param triggerImmediate - Certain events can force immediate action, since browsers like Chrome
   *                                     only allow certain operations in the callback for a user gesture (e.g. like
   *                                     a mouseup to open a window).
   */
  batchEvent( domEvent: Event, batchType: BatchedDOMEventType, callback: BatchedDOMEventCallback, triggerImmediate: boolean ) {
    sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent( 'Input.batchEvent' );
    sceneryLog && sceneryLog.InputEvent && sceneryLog.push();

    // If our display is not interactive, do not respond to any events (but still prevent default)
    if ( this.display.interactive ) {
      this.batchedEvents.push( BatchedDOMEvent.pool.create( domEvent, batchType, callback ) );
      if ( triggerImmediate || !this.batchDOMEvents ) {
        this.fireBatchedEvents();
      }
      // NOTE: If we ever want to Display.updateDisplay() on events, do so here
    }

    // Always preventDefault on touch events, since we don't want mouse events triggered afterwards. See
    // http://www.html5rocks.com/en/mobile/touchandmouse/ for more information.
    // Additionally, IE had some issues with skipping prevent default, see
    // https://github.com/phetsims/scenery/issues/464 for mouse handling.
    if ( !( this.passiveEvents === true ) && ( callback !== this.mouseDown || platform.edge ) ) {
      // We cannot prevent a passive event, so don't try
      domEvent.preventDefault();
    }

    sceneryLog && sceneryLog.InputEvent && sceneryLog.pop();
  }

  /**
   * Fires all of our events that were batched into the batchedEvents array. (scenery-internal)
   */
  fireBatchedEvents() {
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
  clearBatchedEvents() {
    this.batchedEvents.length = 0;
  }

  /**
   * Checks all pointers to see whether they are still "over" the same nodes (trail). If not, it will fire the usual
   * enter/exit events. (scenery-internal)
   */
  validatePointers() {
    this.validatePointersAction.execute();
  }

  /**
   * Removes all non-Mouse pointers from internal tracking. (scenery-internal)
   */
  removeTemporaryPointers() {
    // TODO: Just null this out, instead of creating a fake event?
    const fakeDomEvent = {
      eek: 'This is a fake DOM event created in removeTemporaryPointers(), called from a Scenery exit event. Our attempt to masquerade seems unsuccessful! :('
    } as unknown as Event;

    for ( let i = this.pointers.length - 1; i >= 0; i-- ) {
      const pointer = this.pointers[ i ];
      if ( !( pointer instanceof Mouse ) ) {
        this.pointers.splice( i, 1 );

        // Send exit events. As we can't get a DOM event, we'll send a fake object instead.
        const exitTrail = pointer.trail || new Trail( this.rootNode );
        this.exitEvents( pointer, fakeDomEvent, exitTrail, 0, true );
      }
    }
  }

  /**
   * Hooks up DOM listeners to whatever type of object we are going to listen to. (scenery-internal)
   */
  connectListeners() {
    BrowserEvents.addDisplay( this.display, this.attachToWindow, this.passiveEvents );

    if ( this.display._accessible ) {

      // Add a listener to the root accessible DOM element for each event we want to monitor.
      this.pdomEventListenerMap!.forEach( ( listener, eventName ) => {
        this.display.pdomRootElement!.addEventListener( eventName, listener, this.accessibleEventOptions );
      } );
    }
  }

  /**
   * Removes DOM listeners from whatever type of object we were listening to. (scenery-internal)
   */
  disconnectListeners() {
    BrowserEvents.removeDisplay( this.display, this.attachToWindow, this.passiveEvents );

    if ( this.display._accessible ) {

      // Remove listeners from the root accessible DOM element for each event we want to monitor.
      this.pdomEventListenerMap!.forEach( ( listener, eventName ) => {
        this.display.pdomRootElement!.removeEventListener( eventName, listener, this.accessibleEventOptions );
      } );
    }
  }

  /**
   * Extract a {Vector2} global coordinate point from an arbitrary DOM event. (scenery-internal)
   */
  pointFromEvent( domEvent: MouseEvent | WindowTouch ): Vector2 {
    const position = Vector2.pool.create( domEvent.clientX, domEvent.clientY );
    if ( !this.assumeFullWindow ) {
      const domBounds = this.display.domElement.getBoundingClientRect();

      // TODO: consider totally ignoring any with zero width/height, as we aren't attached to the display?
      // For now, don't offset.
      if ( domBounds.width > 0 && domBounds.height > 0 ) {
        position.subtractXY( domBounds.left, domBounds.top );

        // Detect a scaling of the display here (the client bounding rect having different dimensions from our
        // display), and attempt to compensate.
        // NOTE: We can't handle rotation here.
        if ( domBounds.width !== this.display.width || domBounds.height !== this.display.height ) {
          // TODO: Have code verify the correctness here, and that it's not triggering all the time
          position.x *= this.display.width / domBounds.width;
          position.y *= this.display.height / domBounds.height;
        }
      }
    }
    return position;
  }

  /**
   * Adds a pointer to our list.
   */
  private addPointer( pointer: Pointer ) {
    this.pointers.push( pointer );

    this.pointerAddedEmitter.emit( pointer );
  }

  /**
   * Removes a pointer from our list. If we get future events for it (based on the ID) it will be ignored.
   */
  private removePointer( pointer: Pointer ) {
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

  /**
   * Initializes the Mouse object on the first mouse event (this may never happen on touch devices).
   */
  private initMouse( point: Vector2 ): Mouse {
    const mouse = new Mouse( point );
    this.mouse = mouse;
    this.addPointer( mouse );
    return mouse;
  }

  private ensureMouse( point: Vector2 ): Mouse {
    const mouse = this.mouse;
    if ( mouse ) {
      return mouse;
    }
    else {
      return this.initMouse( point );
    }
  }

  /**
   * Initializes the accessible pointer object on the first pdom event.
   */
  private initPDOMPointer(): PDOMPointer {
    const pdomPointer = new PDOMPointer( this.display );
    this.pdomPointer = pdomPointer;

    this.addPointer( pdomPointer );

    return pdomPointer;
  }

  private ensurePDOMPointer(): PDOMPointer {
    const pdomPointer = this.pdomPointer;
    if ( pdomPointer ) {
      return pdomPointer;
    }
    else {
      return this.initPDOMPointer();
    }
  }

  /**
   * Steps to dispatch a pdom-related event. Before dispatch, the PDOMPointer is initialized if it
   * hasn't been created yet and a userGestureEmitter emits to indicate that a user has begun an interaction.
   */
  private dispatchPDOMEvent<DOMEvent extends Event>( trail: Trail, eventType: string, domEvent: DOMEvent, bubbles: boolean ) {

    // exclude focus and blur events because they can happen with scripting without user input
    if ( PDOMUtils.USER_GESTURE_EVENTS.includes( eventType ) ) {
      Display.userGestureEmitter.emit();
    }

    // This workaround hopefully won't be here forever, see ParallelDOM.setExcludeLabelSiblingFromInput() and https://github.com/phetsims/a11y-research/issues/156
    if ( !( domEvent.target && ( domEvent.target as Element ).hasAttribute( PDOMUtils.DATA_EXCLUDE_FROM_INPUT ) ) ) {

      // If the trail is not pickable, don't dispatch PDOM events to those targets - but we still
      // dispatch with an empty trail to call listeners on the Display and Pointer.
      const canFireListeners = trail.isPickable() || PDOM_UNPICKABLE_EVENTS.includes( eventType );

      if ( !canFireListeners ) {
        trail = new Trail( [] );
      }
      assert && assert( this.pdomPointer );
      this.dispatchEvent<DOMEvent>( trail, eventType, this.pdomPointer!, domEvent, bubbles );
    }
  }

  /**
   * From a DOM Event, get its relatedTarget and map that to the scenery Node. Will return null if relatedTarget
   * is not provided, or if relatedTarget is not under PDOM, or there is no associated Node with trail id on the
   * relatedTarget element. (scenery-internal)
   *
   * @param {Event} domEvent - DOM Event, not a SceneryEvent!
   * @returns {Trail|null}
   */
  getRelatedTargetTrail( domEvent: FocusEvent | MouseEvent ): Trail | null {
    // @ts-ignore TODO what is the related target substitute? Not found on DOM APIs
    const relatedTargetElement = domEvent.relatedTarget || domEvent[ RELATED_TARGET_SUBSTITUTE_KEY ];

    if ( relatedTargetElement && this.isTargetUnderPDOM( relatedTargetElement ) ) {
      const relatedTargetId = this.getRelatedTargetTrailId( domEvent );
      return relatedTargetId ? Trail.fromUniqueId( this.display.rootNode, relatedTargetId ) : null;
    }
    return null;
  }

  /**
   * Get the related target trail ID of the node represented by a DOM element in the accessible PDOM.
   */
  private getRelatedTargetTrailId( domEvent: Event ): string {
    return this.getTrailIdImplementation( domEvent, 'relatedTarget', RELATED_TARGET_SUBSTITUTE_KEY );
  }

  /**
   * Update the PDOMPointer with a new trail from a DOMEvent and return it. For multiple dispatches from a single
   * DOMEvent, this ensures that all will dispatch to the same Trail.
   */
  private updateTrailForPDOMDispatch( domEvent: Event ): Trail {
    return this.ensurePDOMPointer().updateTrail( this.getTrailId( domEvent ) );
  }

  /**
   * Get the trail ID of the node represented by a DOM element who is the target of a DOM Event in the accessible PDOM.
   */
  private getTrailId( domEvent: Event ): string {
    return this.getTrailIdImplementation( domEvent, 'target', TARGET_SUBSTITUTE_KEY );
  }

  private getTrailIdImplementation( domEvent: Event, targetKeyName: string, targetSubstitudeKeyName: string ): string {
    const anyEvent = domEvent as any;
    assert && assert( this.display._accessible, 'Display must be accessible to get trail IDs from PDOMPeers' );
    assert && assert( anyEvent[ targetKeyName ] || anyEvent[ targetSubstitudeKeyName ], 'need a way to get the target' );

    // could be serialized event for phet-io playbacks, see Input.serializeDOMEvent()
    if ( anyEvent[ targetSubstitudeKeyName ] ) {
      assert && assert( anyEvent[ targetSubstitudeKeyName ] instanceof Object );
      return anyEvent[ targetSubstitudeKeyName ].getAttribute( PDOMUtils.DATA_TRAIL_ID );
    }
    else {
      assert && assert( anyEvent[ targetKeyName ] instanceof window.Element );
      return anyEvent[ targetKeyName ].getAttribute( PDOMUtils.DATA_TRAIL_ID );
    }
  }

  /**
   * Triggers a logical mousedown event. (scenery-internal)
   *
   * NOTE: This may also be called from the pointer event handler (pointerDown) or from things like fuzzing or
   * playback. The event may be "faked" for certain purposes.
   */
  mouseDown( id: number, point: Vector2, event: MouseEvent | PointerEvent ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `mouseDown('${id}', ${Input.debugText( point, event )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.mouseDownAction.execute( id, point, event );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical mouseup event. (scenery-internal)
   *
   * NOTE: This may also be called from the pointer event handler (pointerUp) or from things like fuzzing or
   * playback. The event may be "faked" for certain purposes.
   */
  mouseUp( point: Vector2, event: MouseEvent | PointerEvent ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `mouseUp(${Input.debugText( point, event )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.mouseUpAction.execute( point, event );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical mousemove event. (scenery-internal)
   *
   * NOTE: This may also be called from the pointer event handler (pointerMove) or from things like fuzzing or
   * playback. The event may be "faked" for certain purposes.
   */
  mouseMove( point: Vector2, event: MouseEvent | PointerEvent ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `mouseMove(${Input.debugText( point, event )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.mouseMoveAction.execute( point, event );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical mouseover event (this does NOT correspond to the Scenery event, since this is for the display) (scenery-internal)
   */
  mouseOver( point: Vector2, event: MouseEvent | PointerEvent ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `mouseOver(${Input.debugText( point, event )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.mouseOverAction.execute( point, event );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical mouseout event (this does NOT correspond to the Scenery event, since this is for the display) (scenery-internal)
   */
  mouseOut( point: Vector2, event: MouseEvent | PointerEvent ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `mouseOut(${Input.debugText( point, event )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.mouseOutAction.execute( point, event );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical mouse-wheel/scroll event. (scenery-internal)
   */
  wheel( event: WheelEvent ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `wheel(${Input.debugText( null, event )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.wheelScrollAction.execute( event );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical touchstart event. This is called for each touch point in a 'raw' event. (scenery-internal)
   *
   * NOTE: This may also be called from the pointer event handler (pointerDown) or from things like fuzzing or
   * playback. The event may be "faked" for certain purposes.
   */
  touchStart( id: number, point: Vector2, event: TouchEvent | PointerEvent ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `touchStart('${id}',${Input.debugText( point, event )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();

    this.touchStartAction.execute( id, point, event );

    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical touchend event. This is called for each touch point in a 'raw' event. (scenery-internal)
   *
   * NOTE: This may also be called from the pointer event handler (pointerUp) or from things like fuzzing or
   * playback. The event may be "faked" for certain purposes.
   */
  touchEnd( id: number, point: Vector2, event: TouchEvent | PointerEvent ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `touchEnd('${id}',${Input.debugText( point, event )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();

    this.touchEndAction.execute( id, point, event );

    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical touchmove event. This is called for each touch point in a 'raw' event. (scenery-internal)
   *
   * NOTE: This may also be called from the pointer event handler (pointerMove) or from things like fuzzing or
   * playback. The event may be "faked" for certain purposes.
   */
  touchMove( id: number, point: Vector2, event: TouchEvent | PointerEvent ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `touchMove('${id}',${Input.debugText( point, event )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.touchMoveAction.execute( id, point, event );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical touchcancel event. This is called for each touch point in a 'raw' event. (scenery-internal)
   *
   * NOTE: This may also be called from the pointer event handler (pointerCancel) or from things like fuzzing or
   * playback. The event may be "faked" for certain purposes.
   */
  touchCancel( id: number, point: Vector2, event: TouchEvent | PointerEvent ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `touchCancel('${id}',${Input.debugText( point, event )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.touchCancelAction.execute( id, point, event );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical penstart event (e.g. a stylus). This is called for each pen point in a 'raw' event. (scenery-internal)
   *
   * NOTE: This may also be called from the pointer event handler (pointerDown) or from things like fuzzing or
   * playback. The event may be "faked" for certain purposes.
   */
  penStart( id: number, point: Vector2, event: PointerEvent ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `penStart('${id}',${Input.debugText( point, event )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.penStartAction.execute( id, point, event );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical penend event (e.g. a stylus). This is called for each pen point in a 'raw' event. (scenery-internal)
   *
   * NOTE: This may also be called from the pointer event handler (pointerUp) or from things like fuzzing or
   * playback. The event may be "faked" for certain purposes.
   */
  penEnd( id: number, point: Vector2, event: PointerEvent ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `penEnd('${id}',${Input.debugText( point, event )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.penEndAction.execute( id, point, event );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical penmove event (e.g. a stylus). This is called for each pen point in a 'raw' event. (scenery-internal)
   *
   * NOTE: This may also be called from the pointer event handler (pointerMove) or from things like fuzzing or
   * playback. The event may be "faked" for certain purposes.
   */
  penMove( id: number, point: Vector2, event: PointerEvent ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `penMove('${id}',${Input.debugText( point, event )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.penMoveAction.execute( id, point, event );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical pencancel event (e.g. a stylus). This is called for each pen point in a 'raw' event. (scenery-internal)
   *
   * NOTE: This may also be called from the pointer event handler (pointerCancel) or from things like fuzzing or
   * playback. The event may be "faked" for certain purposes.
   */
  penCancel( id: number, point: Vector2, event: PointerEvent ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `penCancel('${id}',${Input.debugText( point, event )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.penCancelAction.execute( id, point, event );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Handles a pointerdown event, forwarding it to the proper logical event. (scenery-internal)
   */
  pointerDown( id: number, type: string, point: Vector2, event: PointerEvent ) {
    // In IE for pointer down events, we want to make sure than the next interactions off the page are sent to
    // this element (it will bubble). See https://github.com/phetsims/scenery/issues/464 and
    // http://news.qooxdoo.org/mouse-capturing.
    const target = this.attachToWindow ? document.body : this.display.domElement;
    if ( target.setPointerCapture && event.pointerId ) {
      // NOTE: This will error out if run on a playback destination, where a pointer with the given ID does not exist.
      target.setPointerCapture( event.pointerId );
    }

    type = this.handleUnknownPointerType( type, id );
    switch( type ) {
      case 'mouse':
        // The actual event afterwards
        this.mouseDown( id, point, event );
        break;
      case 'touch':
        this.touchStart( id, point, event );
        break;
      case 'pen':
        this.penStart( id, point, event );
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
  pointerUp( id: number, type: string, point: Vector2, event: PointerEvent ) {

    // update this outside of the Action executions so that PhET-iO event playback does not override it
    this.upTimeStamp = event.timeStamp;

    type = this.handleUnknownPointerType( type, id );
    switch( type ) {
      case 'mouse':
        this.mouseUp( point, event );
        break;
      case 'touch':
        this.touchEnd( id, point, event );
        break;
      case 'pen':
        this.penEnd( id, point, event );
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
  pointerCancel( id: number, type: string, point: Vector2, event: PointerEvent ) {
    type = this.handleUnknownPointerType( type, id );
    switch( type ) {
      case 'mouse':
        if ( console && console.log ) {
          console.log( 'WARNING: Pointer mouse cancel was received' );
        }
        break;
      case 'touch':
        this.touchCancel( id, point, event );
        break;
      case 'pen':
        this.penCancel( id, point, event );
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
  gotPointerCapture( id: number, type: string, point: Vector2, event: Event ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `gotPointerCapture('${id}',${Input.debugText( null, event )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.gotPointerCaptureAction.execute( id, event );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Handles a lostpointercapture event, forwarding it to the proper logical event. (scenery-internal)
   */
  lostPointerCapture( id: number, type: string, point: Vector2, event: Event ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `lostPointerCapture('${id}',${Input.debugText( null, event )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.lostPointerCaptureAction.execute( id, event );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Handles a pointermove event, forwarding it to the proper logical event. (scenery-internal)
   */
  pointerMove( id: number, type: string, point: Vector2, event: PointerEvent ) {
    type = this.handleUnknownPointerType( type, id );
    switch( type ) {
      case 'mouse':
        this.mouseMove( point, event );
        break;
      case 'touch':
        this.touchMove( id, point, event );
        break;
      case 'pen':
        this.penMove( id, point, event );
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
  pointerOver( id: number, type: string, point: Vector2, event: PointerEvent ) {
    // TODO: accumulate mouse/touch info in the object if needed?
    // TODO: do we want to branch change on these types of events?
  }

  /**
   * Handles a pointerout event, forwarding it to the proper logical event. (scenery-internal)
   */
  pointerOut( id: number, type: string, point: Vector2, event: PointerEvent ) {
    // TODO: accumulate mouse/touch info in the object if needed?
    // TODO: do we want to branch change on these types of events?
  }

  /**
   * Handles a pointerenter event, forwarding it to the proper logical event. (scenery-internal)
   */
  pointerEnter( id: number, type: string, point: Vector2, event: PointerEvent ) {
    // TODO: accumulate mouse/touch info in the object if needed?
    // TODO: do we want to branch change on these types of events?
  }

  /**
   * Handles a pointerleave event, forwarding it to the proper logical event. (scenery-internal)
   */
  pointerLeave( id: number, type: string, point: Vector2, event: PointerEvent ) {
    // TODO: accumulate mouse/touch info in the object if needed?
    // TODO: do we want to branch change on these types of events?
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
  private upEvent<DOMEvent extends Event>( pointer: Pointer, event: DOMEvent, pointChanged: boolean ) {

    // if the event target is within the PDOM the AT is sending a fake pointer event to the document - do not
    // dispatch this since the PDOM should only handle Input.PDOM_EVENT_TYPES, and all other pointer input should
    // go through the Display div. Otherwise, activation will be duplicated when we handle pointer and PDOM events
    if ( this.isTargetUnderPDOM( event.target as HTMLElement ) ) {
      return;
    }

    sceneryLog && sceneryLog.Input && sceneryLog.Input( `upEvent ${pointer.toString()} changed:${pointChanged}` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();

    assert && assert( pointer instanceof Pointer );
    assert && assert( typeof pointChanged === 'boolean' );

    // We'll use this trail for the entire dispatch of this event.
    const eventTrail = this.branchChangeEvents<DOMEvent>( pointer, event, pointChanged );

    this.dispatchEvent<DOMEvent>( eventTrail, 'up', pointer, event, true );

    // touch pointers are transient, so fire exit/out to the trail afterwards
    if ( pointer.isTouchLike() ) {
      this.exitEvents<DOMEvent>( pointer, event, eventTrail, 0, true );
    }

    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Called for each logical "down" event, for any pointer type.
   */
  private downEvent<DOMEvent extends Event>( pointer: Pointer, event: DOMEvent, pointChanged: boolean ) {

    // if the event target is within the PDOM the AT is sending a fake pointer event to the document - do not
    // dispatch this since the PDOM should only handle Input.PDOM_EVENT_TYPES, and all other pointer input should
    // go through the Display div. Otherwise, activation will be duplicated when we handle pointer and PDOM events
    if ( this.isTargetUnderPDOM( event.target as HTMLElement ) ) {
      return;
    }

    sceneryLog && sceneryLog.Input && sceneryLog.Input( `downEvent ${pointer.toString()} changed:${pointChanged}` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();

    assert && assert( pointer instanceof Pointer );
    assert && assert( typeof pointChanged === 'boolean' );

    // We'll use this trail for the entire dispatch of this event.
    const eventTrail = this.branchChangeEvents<DOMEvent>( pointer, event, pointChanged );

    this.dispatchEvent<DOMEvent>( eventTrail, 'down', pointer, event, true );

    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Called for each logical "move" event, for any pointer type.
   */
  private moveEvent<DOMEvent extends Event>( pointer: Pointer, event: DOMEvent ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `moveEvent ${pointer.toString()}` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();

    assert && assert( pointer instanceof Pointer );

    // Always treat move events as "point changed"
    this.branchChangeEvents<DOMEvent>( pointer, event, true );

    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Called for each logical "cancel" event, for any pointer type.
   */
  private cancelEvent<DOMEvent extends Event>( pointer: Pointer, event: DOMEvent, pointChanged: boolean ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `cancelEvent ${pointer.toString()} changed:${pointChanged}` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();

    assert && assert( pointer instanceof Pointer );
    assert && assert( typeof pointChanged === 'boolean' );

    // We'll use this trail for the entire dispatch of this event.
    const eventTrail = this.branchChangeEvents<DOMEvent>( pointer, event, pointChanged );

    this.dispatchEvent<DOMEvent>( eventTrail, 'cancel', pointer, event, true );

    // touch pointers are transient, so fire exit/out to the trail afterwards
    if ( pointer.isTouchLike() ) {
      this.exitEvents<DOMEvent>( pointer, event, eventTrail, 0, true );
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
   * @param event
   * @param sendMove - Whether to send move events
   * @returns - The current trail of the pointer
   */
  private branchChangeEvents<DOMEvent extends Event>( pointer: Pointer, event: DOMEvent | null, sendMove: boolean ): Trail {
    sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent(
      `branchChangeEvents: ${pointer.toString()} sendMove:${sendMove}` );
    sceneryLog && sceneryLog.InputEvent && sceneryLog.push();

    assert && assert( pointer instanceof Pointer );
    assert && assert( typeof sendMove === 'boolean' );

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
      this.dispatchEvent<DOMEvent>( trail, 'move', pointer, event, true );
    }

    // We want to approximately mimic http://www.w3.org/TR/DOM-Level-3-Events/#events-mouseevent-event-order
    this.exitEvents<DOMEvent>( pointer, event, oldInputEnabledTrail, branchInputEnabledIndex, lastInputEnabledNodeChanged );
    this.enterEvents<DOMEvent>( pointer, event, inputEnabledTrail, branchInputEnabledIndex, lastInputEnabledNodeChanged );

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
  private enterEvents<DOMEvent extends Event>( pointer: Pointer, event: DOMEvent | null, trail: Trail, branchIndex: number, lastNodeChanged: boolean ) {
    if ( lastNodeChanged ) {
      this.dispatchEvent<DOMEvent>( trail, 'over', pointer, event, true, true );
    }

    for ( let i = branchIndex; i < trail.length; i++ ) {
      this.dispatchEvent<DOMEvent>( trail.slice( 0, i + 1 ), 'enter', pointer, event, false );
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
  private exitEvents<DOMEvent extends Event>( pointer: Pointer, event: DOMEvent | null, trail: Trail, branchIndex: number, lastNodeChanged: boolean ) {
    for ( let i = trail.length - 1; i >= branchIndex; i-- ) {
      this.dispatchEvent<DOMEvent>( trail.slice( 0, i + 1 ), 'exit', pointer, event, false, true );
    }

    if ( lastNodeChanged ) {
      this.dispatchEvent<DOMEvent>( trail, 'out', pointer, event, true );
    }
  }

  /**
   * Dispatch to all nodes in the Trail, optionally bubbling down from the leaf to the root.
   *
   * @param trail
   * @param type
   * @param pointer
   * @param event
   * @param bubbles - If bubbles is false, the event is only dispatched to the leaf node of the trail.
   * @param fireOnInputDisabled - Whether to fire this event even if nodes have inputEnabled:false
   */
  private dispatchEvent<DOMEvent extends Event>( trail: Trail, type: string, pointer: Pointer, event: DOMEvent | null, bubbles: boolean, fireOnInputDisabled: boolean = false ) {
    sceneryLog && sceneryLog.EventDispatch && sceneryLog.EventDispatch(
      `${type} trail:${trail.toString()} pointer:${pointer.toString()} at ${pointer.point ? pointer.point.toString() : 'null'}` );
    sceneryLog && sceneryLog.EventDispatch && sceneryLog.push();

    assert && assert( trail, 'Falsy trail for dispatchEvent' );

    sceneryLog && sceneryLog.EventPath && sceneryLog.EventPath( `${type} ${trail.toPathString()}` );

    // NOTE: event is not immutable, as its currentTarget changes
    const inputEvent = new SceneryEvent<DOMEvent>( trail, type, pointer, event );

    // first run through the pointer's listeners to see if one of them will handle the event
    this.dispatchToListeners<DOMEvent>( pointer, pointer.getListeners(), type, inputEvent );

    // if not yet handled, run through the trail in order to see if one of them will handle the event
    // at the base of the trail should be the scene node, so the scene will be notified last
    this.dispatchToTargets<DOMEvent>( trail, type, pointer, inputEvent, bubbles, fireOnInputDisabled );

    // Notify input listeners on the Display
    this.dispatchToListeners<DOMEvent>( pointer, this.display.getInputListeners(), type, inputEvent );

    // Notify input listeners to any Display
    if ( Display.inputListeners.length ) {
      this.dispatchToListeners<DOMEvent>( pointer, Display.inputListeners.slice(), type, inputEvent );
    }

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
  private dispatchToListeners<DOMEvent extends Event>( pointer: Pointer, listeners: IInputListener[], type: string, inputEvent: SceneryEvent<DOMEvent> ) {
    assert && assert( inputEvent instanceof SceneryEvent );

    if ( inputEvent.handled ) {
      return;
    }

    const specificType = pointer.type + type; // e.g. mouseup, touchup

    for ( let i = 0; i < listeners.length; i++ ) {
      const listener = listeners[ i ];

      if ( !inputEvent.aborted && listener[ specificType as keyof IInputListener ] ) {
        sceneryLog && sceneryLog.EventDispatch && sceneryLog.EventDispatch( specificType );
        sceneryLog && sceneryLog.EventDispatch && sceneryLog.push();

        ( listener[ specificType as keyof IInputListener ] as SceneryListenerFunction<DOMEvent> )( inputEvent );

        sceneryLog && sceneryLog.EventDispatch && sceneryLog.pop();
      }

      if ( !inputEvent.aborted && listener[ type as keyof IInputListener ] ) {
        sceneryLog && sceneryLog.EventDispatch && sceneryLog.EventDispatch( type );
        sceneryLog && sceneryLog.EventDispatch && sceneryLog.push();

        ( listener[ type as keyof IInputListener ] as SceneryListenerFunction<DOMEvent> )( inputEvent );

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
  private dispatchToTargets<DOMEvent extends Event>( trail: Trail, type: string, pointer: Pointer, inputEvent: SceneryEvent<DOMEvent>, bubbles: boolean, fireOnInputDisabled: boolean = false ) {
    assert && assert( inputEvent instanceof SceneryEvent );

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
   * Returns true if the Display is accessible and the element is a descendant of the Display PDOM.
   */
  private isTargetUnderPDOM( element: HTMLElement ): boolean {
    return this.display._accessible && this.display.pdomRootElement!.contains( element );
  }

  /**
   * Saves the main information we care about from a DOM `Event` into a JSON-like structure. To support
   * polymorphism, all supported DOM event keys that scenery uses will always be included in this serialization. If
   * the particular Event interface for the instance being serialized doesn't have a certain property, then it will be
   * set as `null`. See domEventPropertiesToSerialize for the full list of supported Event properties.
   *
   * @returns {Object} - see domEventPropertiesToSerialize for list keys that are serialized
   */
  static serializeDomEvent( domEvent: Event ): any {
    const entries = {} as any;

    domEventPropertiesToSerialize.forEach( property => {

        const domEventProperty = domEvent[ property as keyof Event ] as any;

        // We serialize many Event APIs into a single object, so be graceful if properties don't exist.
        if ( domEventProperty === undefined || domEventProperty === null ) {
          entries[ property ] = null;
        }

        // stringifying dom event object properties can cause circular references, so we avoid that completely
        else if ( property === 'touches' || property === 'targetTouches' || property === 'changedTouches' ) {

          const touchArray = [];
          for ( let i = 0; i < domEventProperty.length; i++ ) {

            // According to spec (http://www.w3.org/TR/touch-events/), this is not an Array, but a TouchList. In practice
            // the phet-io team found that chrome and safari, along with downstream "playback" phet-io sims, use an Array.
            // So we need to support both APIs.
            const touch = ( domEventProperty.item && typeof domEventProperty.item === 'function' ) ?
                          domEventProperty.item( i ) :
                          domEventProperty[ i ];

            touchArray.push( Input.serializeDomEvent( touch ) );
          }
          entries[ property ] = touchArray;
        }

        else if ( EVENT_KEY_VALUES_AS_ELEMENTS.includes( property ) && typeof domEventProperty.getAttribute === 'function' &&

                  // If false, then this target isn't a PDOM element, so we can skip this serialization
                  domEventProperty.hasAttribute( PDOMUtils.DATA_TRAIL_ID ) ) {

          // If the target came from the accessibility PDOM, then we want to store the Node trail id of where it came from.

          entries[ property ] = {};
          entries[ property ][ PDOMUtils.DATA_TRAIL_ID ] = domEventProperty.getAttribute( PDOMUtils.DATA_TRAIL_ID );
        }
        else {

          // Parse to get rid of functions and circular references.
          entries[ property ] = ( ( typeof domEventProperty === 'object' ) ? {} : JSON.parse( JSON.stringify( domEventProperty ) ) );
        }
      }
    );
    return entries;
  }

  /**
   * From a serialized dom event, return a recreated window.Event (scenery-internal)
   */
  static deserializeDomEvent( eventObject: any ): Event {
    const domEvent: Event = new window.Event( 'Event' );
    for ( const key in eventObject ) {

      // `type` is readonly, so don't try to set it.
      if ( eventObject.hasOwnProperty( key ) && key !== 'type' ) {

        // Special case for target since we can't set that read-only property. Instead use a substitute key.
        if ( key === 'target' ) {
          // @ts-ignore
          domEvent[ TARGET_SUBSTITUTE_KEY ] = _.clone( eventObject[ key ] ) || {};

          // TODO: only needed until https://github.com/phetsims/scenery/issues/1296 is complete, double check on getTrailIdImplementation() too
          // @ts-ignore
          domEvent[ TARGET_SUBSTITUTE_KEY ].getAttribute = function( key ) {
            return this[ key ];
          };
        }
        else if ( key === 'relatedTarget' ) {
          if ( eventObject[ key ] ) {

            const htmlElement = document.getElementById( eventObject[ key ][ PDOMUtils.DATA_TRAIL_ID ] );
            assert && assert( htmlElement, 'cannot deserialize event when related target is not in the DOM.' );
            // @ts-ignore
            domEvent[ RELATED_TARGET_SUBSTITUTE_KEY ] = htmlElement;
          }
        }
        else {
          // @ts-ignore
          domEvent[ key ] = eventObject[ key ];
        }
      }
    }
    return domEvent;
  }

  /**
   * Convenience function for logging out a point/event combination.
   *
   * @param point - Not logged if null
   * @param domEvent
   */
  private static debugText( point: Vector2 | null, domEvent: Event ) {
    let result = `${domEvent.timeStamp} ${domEvent.type}`;
    if ( point !== null ) {
      result = `${point.x},${point.y} ${result}`;
    }
    return result;
  }

  /**
   * Maps the current MS pointer types onto the pointer spec. (scenery-internal)
   */
  static msPointerType( event: PointerEvent ): string {
    // @ts-ignore -- legacy API
    if ( event.pointerType === window.MSPointerEvent.MSPOINTER_TYPE_TOUCH ) {
      return 'touch';
    }
    // @ts-ignore -- legacy API
    else if ( event.pointerType === window.MSPointerEvent.MSPOINTER_TYPE_PEN ) {
      return 'pen';
    }
    // @ts-ignore -- legacy API
    else if ( event.pointerType === window.MSPointerEvent.MSPOINTER_TYPE_MOUSE ) {
      return 'mouse';
    }
    else {
      return event.pointerType; // hope for the best
    }
  }
}

scenery.register( 'Input', Input );
export default Input;
export type { InputOptions };
