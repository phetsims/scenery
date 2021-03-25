// Copyright 2013-2020, University of Colorado Boulder

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
import Vector2 from '../../../dot/js/Vector2.js';
import Vector2IO from '../../../dot/js/Vector2IO.js';
import cleanArray from '../../../phet-core/js/cleanArray.js';
import merge from '../../../phet-core/js/merge.js';
import platform from '../../../phet-core/js/platform.js';
import EventType from '../../../tandem/js/EventType.js';
import Tandem from '../../../tandem/js/Tandem.js';
import NullableIO from '../../../tandem/js/types/NullableIO.js';
import NumberIO from '../../../tandem/js/types/NumberIO.js';
import KeyboardUtils from '../accessibility/KeyboardUtils.js';
import PDOMUtils from '../accessibility/pdom/PDOMUtils.js';
import Display from '../display/Display.js';
import scenery from '../scenery.js';
import Features from '../util/Features.js';
import FullScreen from '../util/FullScreen.js';
import Trail from '../util/Trail.js';
import BatchedDOMEvent from './BatchedDOMEvent.js';
import BrowserEvents from './BrowserEvents.js';
import EventIO from './EventIO.js';
import Mouse from './Mouse.js';
import PDOMPointer from './PDOMPointer.js';
import Pen from './Pen.js';
import Pointer from './Pointer.js';
import SceneryEvent from './SceneryEvent.js';
import Touch from './Touch.js';

// Object literal makes it easy to check for the existence of an attribute (compared to [].indexOf()>=0)
const domEventPropertiesToSerialize = {
  type: true,
  button: true, keyCode: true,
  deltaX: true, deltaY: true, deltaZ: true, deltaMode: true, pointerId: true,
  pointerType: true, charCode: true, which: true, clientX: true, clientY: true, changedTouches: true,
  target: true,
  ctrlKey: true, shiftKey: true, altKey: true, metaKey: true
};

const TARGET_SUBSTITUTE_KEY = 'targetSubstitute';

/**
 * An input controller for a specific Display.
 * @constructor
 *
 * @param {Display} display
 * @param {boolean} attachToWindow - Whether to add listeners to the window (instead of the Display's domElement).
 * @param {boolean} batchDOMEvents - If true, most event types will be batched until otherwise triggered.
 * @param {boolean} assumeFullWindow - We can optimize certain things like computing points if we know the display
 *                                     fills the entire window.
 * @param {boolean|null} passiveEvents - See Display's documentation (controls the presence of the passive flag for
 *                                       events, which has some advanced considerations).
 *
 * @param {Object} [options]
 */
class Input {
  constructor( display, attachToWindow, batchDOMEvents, assumeFullWindow, passiveEvents, options ) {
    assert && assert( display instanceof Display );
    assert && assert( typeof attachToWindow === 'boolean' );
    assert && assert( typeof batchDOMEvents === 'boolean' );
    assert && assert( typeof assumeFullWindow === 'boolean' );

    options = merge( {
      tandem: Tandem.OPTIONAL
    }, options );

    // @public {Display}
    this.display = display;

    // @public {Node}
    this.rootNode = display.rootNode;

    // @public {boolean}
    this.attachToWindow = attachToWindow;
    this.batchDOMEvents = batchDOMEvents;
    this.assumeFullWindow = assumeFullWindow;

    // @public {boolean|null}
    this.passiveEvents = passiveEvents;

    // @private {Array.<BatchedDOMEvent}>
    this.batchedEvents = [];

    // @public {PDOMPointer|null} - Pointer for accessibility, only created lazily on first a11y event.
    this.a11yPointer = null;

    // @public {Mouse|null} - Pointer for mouse, only created lazily on first mouse event, so no mouse is allocated on.
    // tablets.
    this.mouse = null;

    // @public {Array.<Pointer>} - All active pointers.
    this.pointers = [];

    // TODO: replace this with an Emitter
    this.pointerAddedListeners = [];

    // @public {boolean} - Whether we are currently firing events. We need to track this to handle re-entrant cases
    // like https://github.com/phetsims/balloons-and-static-electricity/issues/406.
    this.currentlyFiringEvents = false;

    ////////////////////////////////////////////////////
    // Declare the Actions that send scenery input events to the PhET-iO data stream.  Note they use the default value
    // of phetioReadOnly false, in case a client wants to synthesize events.

    // @private {Action} - Emits pointer validation to the input stream for playback
    // This is a high frequency event that is necessary for reproducible playbacks
    this.validatePointersAction = new Action( () => {
      let i = this.pointers.length;
      while ( i-- ) {
        const pointer = this.pointers[ i ];
        if ( pointer.point ) {
          this.branchChangeEvents( pointer, pointer.lastDOMEvent, false );
        }
      }
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'validatePointersAction' ),
      phetioHighFrequency: true
    } );

    // @private {Action} - Emits to the PhET-iO data stream.
    this.mouseUpAction = new Action( ( point, event ) => {
      if ( !this.mouse ) { this.initMouse(); }
      const pointChanged = this.mouse.up( point, event );
      this.mouse.id = null;
      this.upEvent( this.mouse, event, pointChanged );
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'mouseUpAction' ),
      parameters: [
        { name: 'point', phetioType: Vector2IO },
        { name: 'event', phetioType: EventIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when a mouse button is released.'
    } );

    // @private {Action} - Emits to the PhET-iO data stream.
    this.mouseDownAction = new Action( ( id, point, event ) => {
      if ( !this.mouse ) { this.initMouse(); }
      this.mouse.id = id;
      const pointChanged = this.mouse.down( point, event );
      this.downEvent( this.mouse, event, pointChanged );
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'mouseDownAction' ),
      parameters: [
        { name: 'id', phetioType: NullableIO( NumberIO ) },
        { name: 'point', phetioType: Vector2IO },
        { name: 'event', phetioType: EventIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when a mouse button is pressed.'
    } );

    // @private {Action} - Emits to the PhET-iO data stream.
    this.mouseMoveAction = new Action( ( point, event ) => {
      if ( !this.mouse ) { this.initMouse(); }
      this.mouse.move( point, event );
      this.moveEvent( this.mouse, event );
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'mouseMoveAction' ),
      parameters: [
        { name: 'point', phetioType: Vector2IO },
        { name: 'event', phetioType: EventIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when the mouse is moved.',
      phetioHighFrequency: true
    } );

    // @private {Action} - Emits to the PhET-iO data stream.
    this.mouseOverAction = new Action( ( point, event ) => {
      if ( !this.mouse ) { this.initMouse(); }
      this.mouse.over( point, event );
      // TODO: how to handle mouse-over (and log it)... are we changing the pointer.point without a branch change?
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'mouseOverAction' ),
      parameters: [
        { name: 'point', phetioType: Vector2IO },
        { name: 'event', phetioType: EventIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when the mouse is moved while on the sim.'
    } );

    // @private {Action} - Emits to the PhET-iO data stream.
    this.mouseOutAction = new Action( ( point, event ) => {
      if ( !this.mouse ) { this.initMouse(); }
      this.mouse.out( point, event );
      // TODO: how to handle mouse-out (and log it)... are we changing the pointer.point without a branch change?
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'mouseOutAction' ),
      parameters: [
        { name: 'point', phetioType: Vector2IO },
        { name: 'event', phetioType: EventIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when the mouse moves out of the display.'
    } );

    // @private {Action} - Emits to the PhET-iO data stream.
    this.wheelScrollAction = new Action( event => {
      if ( !this.mouse ) { this.initMouse(); }
      this.mouse.wheel( event );

      // don't send mouse-wheel events if we don't yet have a mouse location!
      // TODO: Can we set the mouse location based on the wheel event?
      if ( this.mouse.point ) {
        const trail = this.rootNode.trailUnderPointer( this.mouse ) || new Trail( this.rootNode );
        this.dispatchEvent( trail, 'wheel', this.mouse, event, true );
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

    // @private {Action} - Emits to the PhET-iO data stream.
    this.touchStartAction = new Action( ( id, point, event ) => {
      const touch = new Touch( id, point, event );
      this.addPointer( touch );
      this.downEvent( touch, event, false );
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'touchStartAction' ),
      parameters: [
        { name: 'id', phetioType: NumberIO },
        { name: 'point', phetioType: Vector2IO },
        { name: 'event', phetioType: EventIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when a touch begins.'
    } );

    // @private {Action} - Emits to the PhET-iO data stream.
    this.touchEndAction = new Action( ( id, point, event ) => {
      const touch = this.findPointerById( id );
      if ( touch ) {
        const pointChanged = touch.end( point, event );
        this.upEvent( touch, event, pointChanged );
        this.removePointer( touch );
      }
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'touchEndAction' ),
      parameters: [
        { name: 'id', phetioType: NumberIO },
        { name: 'point', phetioType: Vector2IO },
        { name: 'event', phetioType: EventIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when a touch ends.'
    } );

    // @private {Action} - Emits to the PhET-iO data stream.
    this.touchMoveAction = new Action( ( id, point, event ) => {
      const touch = this.findPointerById( id );
      if ( touch ) {
        touch.move( point, event );
        this.moveEvent( touch, event );
      }
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'touchMoveAction' ),
      parameters: [
        { name: 'id', phetioType: NumberIO },
        { name: 'point', phetioType: Vector2IO },
        { name: 'event', phetioType: EventIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when a touch moves.',
      phetioHighFrequency: true
    } );

    // @private {Action} - Emits to the PhET-iO data stream.
    this.touchCancelAction = new Action( ( id, point, event ) => {
      const touch = this.findPointerById( id );
      if ( touch ) {
        const pointChanged = touch.cancel( point, event );
        this.cancelEvent( touch, event, pointChanged );
        this.removePointer( touch );
      }
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'touchCancelAction' ),
      parameters: [
        { name: 'id', phetioType: NumberIO },
        { name: 'point', phetioType: Vector2IO },
        { name: 'event', phetioType: EventIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when a touch is canceled.'
    } );

    // @private {Action} - Emits to the PhET-iO data stream.
    this.penStartAction = new Action( ( id, point, event ) => {
      const pen = new Pen( id, point, event );
      this.addPointer( pen );
      this.downEvent( pen, event, false );
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'penStartAction' ),
      parameters: [
        { name: 'id', phetioType: NumberIO },
        { name: 'point', phetioType: Vector2IO },
        { name: 'event', phetioType: EventIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when a pen touches the screen.'
    } );

    // @private {Action} - Emits to the PhET-iO data stream.
    this.penEndAction = new Action( ( id, point, event ) => {
      const pen = this.findPointerById( id );
      if ( pen ) {
        const pointChanged = pen.end( point, event );
        this.upEvent( pen, event, pointChanged );
        this.removePointer( pen );
      }
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'penEndAction' ),
      parameters: [
        { name: 'id', phetioType: NumberIO },
        { name: 'point', phetioType: Vector2IO },
        { name: 'event', phetioType: EventIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when a pen is lifted.'
    } );

    // @private {Action} - Emits to the PhET-iO data stream.
    this.penMoveAction = new Action( ( id, point, event ) => {
      const pen = this.findPointerById( id );
      if ( pen ) {
        pen.move( point, event );
        this.moveEvent( pen, event );
      }
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'penMoveAction' ),
      parameters: [
        { name: 'id', phetioType: NumberIO },
        { name: 'point', phetioType: Vector2IO },
        { name: 'event', phetioType: EventIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when a pen is moved.',
      phetioHighFrequency: true
    } );

    // @private {Action} - Emits to the PhET-iO data stream.
    this.penCancelAction = new Action( ( id, point, event ) => {
      const pen = this.findPointerById( id );
      if ( pen ) {
        const pointChanged = pen.cancel( point, event );
        this.cancelEvent( pen, event, pointChanged );
        this.removePointer( pen );
      }
    }, {
      phetioPlayback: true,
      tandem: options.tandem.createTandem( 'penCancelAction' ),
      parameters: [
        { name: 'id', phetioType: NumberIO },
        { name: 'point', phetioType: Vector2IO },
        { name: 'event', phetioType: EventIO }
      ],
      phetioEventType: EventType.USER,
      phetioDocumentation: 'Emits when a pen is canceled.'
    } );

    // @private {Action} - Emits to the PhET-iO data stream.
    this.gotPointerCaptureAction = new Action( ( id, event ) => {
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

    // @private {Action} - Emits to the PhET-iO data stream.
    this.lostPointerCaptureAction = new Action( ( id, event ) => {
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

      // In IE11, the focusin event can be sent twice since we often have to restore focus to event.relatedTarget
      // after calling focusout callbacks. So this flag is set to prevent focusin callbacks from firing twice
      // when that happens. See https://github.com/phetsims/scenery/issues/925
      let ieBlockCallbacks = false;

      // @private
      this.focusinAction = new Action( event => {

        // ignore any focusout callbacks if they are initiated due to implementation details in PDOM manipulation
        if ( this.display.blockFocusCallbacks ) {
          return;
        }

        if ( ieBlockCallbacks ) {
          ieBlockCallbacks = false;
          return;
        }

        sceneryLog && sceneryLog.Input && sceneryLog.Input( 'focusin(' + Input.debugText( null, event ) + ');' );
        sceneryLog && sceneryLog.Input && sceneryLog.push();

        const trail = this.updateTrailForPDOMDispatch( event );
        this.dispatchA11yEvent( trail, 'focus', event, false );
        this.dispatchA11yEvent( trail, 'focusin', event, true );

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

      // @private
      this.focusoutAction = new Action( event => {

        // ignore any focusout callbacks if they are initiated due to implementation details in PDOM manipulation
        if ( this.display.blockFocusCallbacks ) {
          return;
        }

        sceneryLog && sceneryLog.Input && sceneryLog.Input( 'focusOut(' + Input.debugText( null, event ) + ');' );
        sceneryLog && sceneryLog.Input && sceneryLog.push();

        // recompute the trail on focusout if necessary - since a blur/focusout may have been initiated from a
        // focus/focusin listener, it is possible that focusout was called more than once before focusin is called on the
        // next active element, see https://github.com/phetsims/scenery/issues/898
        if ( !this.a11yPointer ) { this.initA11yPointer(); }
        this.a11yPointer.invalidateTrail( this.getTrailId( event ) );

        const trail = this.updateTrailForPDOMDispatch( event );
        this.dispatchA11yEvent( trail, 'blur', event, false );
        this.dispatchA11yEvent( trail, 'focusout', event, true );

        // clear the trail to make sure that our assertions aren't testing a stale trail, do this before
        // focusing event.relatedTarget below so that trail isn't cleared after focus
        this.a11yPointer.trail = null;

        // If there exists an event.relatedTarget, user is moving to the next item with "tab" like navigation and
        // event.relatedTarget is the element focus is moving to. If the relatedTarget element is removed from the
        // document then added back in during focusout callbacks (which is done in AccessibilityTree operations),
        // some browsers (IE11 and Safari) will fail to focus the element. So we manually focus the relatedTarget
        // so that focus isn't lost. Skipped if focusout callbacks sent focus to an element other than the
        // relatedTarget, or if callbacks made relatedTarget unfocusable.
        //
        // Focus is set with DOM API to avoid the performance hit of looking up the Node from trail id.
        if ( event.relatedTarget ) {
          const focusMovedInCallbacks = this.isTargetUnderPDOM( document.activeElement );
          const targetFocusable = PDOMUtils.isElementFocusable( event.relatedTarget );
          if ( targetFocusable && !focusMovedInCallbacks ) {
            if ( platform.ie ) {
              ieBlockCallbacks = true;
            }
            event.relatedTarget.focus();
          }
        }

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

      // @private
      this.clickAction = new Action( event => {
        sceneryLog && sceneryLog.Input && sceneryLog.Input( 'click(' + Input.debugText( null, event ) + ');' );
        sceneryLog && sceneryLog.Input && sceneryLog.push();

        const trail = this.updateTrailForPDOMDispatch( event );
        this.dispatchA11yEvent( trail, 'click', event, true );

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

      // @private
      this.inputAction = new Action( event => {
        sceneryLog && sceneryLog.Input && sceneryLog.Input( 'input(' + Input.debugText( null, event ) + ');' );
        sceneryLog && sceneryLog.Input && sceneryLog.push();

        const trail = this.updateTrailForPDOMDispatch( event );
        this.dispatchA11yEvent( trail, 'input', event, true );

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

      // @private
      this.changeAction = new Action( event => {
        sceneryLog && sceneryLog.Input && sceneryLog.Input( 'change(' + Input.debugText( null, event ) + ');' );
        sceneryLog && sceneryLog.Input && sceneryLog.push();

        const trail = this.updateTrailForPDOMDispatch( event );
        this.dispatchA11yEvent( trail, 'change', event, true );

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

      // @private
      this.keydownAction = new Action( event => {
        sceneryLog && sceneryLog.Input && sceneryLog.Input( 'keydown(' + Input.debugText( null, event ) + ');' );
        sceneryLog && sceneryLog.Input && sceneryLog.push();

        const trail = this.updateTrailForPDOMDispatch( event );
        this.dispatchA11yEvent( trail, 'keydown', event, true );

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

      // @private
      this.keyupAction = new Action( event => {
        sceneryLog && sceneryLog.Input && sceneryLog.Input( 'keyup(' + Input.debugText( null, event ) + ');' );
        sceneryLog && sceneryLog.Input && sceneryLog.push();

        const trail = this.updateTrailForPDOMDispatch( event );
        this.dispatchA11yEvent( trail, 'keyup', event, true );

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

      // same event options for all DOM listeners
      const accessibleEventOptions = Features.passive ? { useCapture: false, passive: false } : false;

      // Add a listener to the root accessible DOM element for each event we want to monitor.
      PDOMUtils.DOM_EVENTS.forEach( eventName => {

        const actionName = eventName + 'Action';
        assert && assert( this[ actionName ], `action not defined on Input: ${actionName}` );

        // These exist for the lifetime of the display, and need not be disposed.
        this.display.accessibleDOMElement.addEventListener( eventName, event => {

          sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent( `Input.${eventName}FromBrowser` );
          sceneryLog && sceneryLog.InputEvent && sceneryLog.push();

          // Create the a11yPointer lazily
          this[ actionName ].execute( event );

          sceneryLog && sceneryLog.InputEvent && sceneryLog.pop();
        }, accessibleEventOptions );
      } );

      // Add a listener to the document body that will capture any keydown for a11y before focus is in this display.
      document.body.addEventListener( 'keydown', this.handleDocumentKeydown.bind( this ) );
    }
  }

  /**
   * Interrupts any input actions that are currently taking place (should stop drags, etc.)
   * @public
   */
  interruptPointers() {
    _.each( this.pointers, pointer => {
      pointer.interruptAll();
    } );
  }

  /**
   * Called to batch a raw DOM event (which may be immediately fired, depending on the settings).
   * @public (scenery-internal)
   *
   * @param {Event} domEvent
   * @param {number} batchType - See BatchedDOMEvent's "enumeration" - TODO: use an actual enumeration
   * @param {function} callback - Parameter types defined by the batchType. See BatchedDOMEvent for details
   * @param {boolean} triggerImmediate - Certain events can force immediate action, since browsers like Chrome
   *                                     only allow certain operations in the callback for a user gesture (e.g. like
   *                                     a mouseup to open a window).
   */
  batchEvent( domEvent, batchType, callback, triggerImmediate ) {
    sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent( 'Input.batchEvent' );
    sceneryLog && sceneryLog.InputEvent && sceneryLog.push();

    // If our display is not interactive, do not respond to any events (but still prevent default)
    if ( this.display.interactive ) {
      this.batchedEvents.push( BatchedDOMEvent.createFromPool( domEvent, batchType, callback ) );
      if ( triggerImmediate || !this.batchDOMEvents ) {
        this.fireBatchedEvents();
      }
      // NOTE: If we ever want to Display.updateDisplay() on events, do so here
    }

    // Always preventDefault on touch events, since we don't want mouse events triggered afterwards. See
    // http://www.html5rocks.com/en/mobile/touchandmouse/ for more information.
    // Additionally, IE had some issues with skipping prevent default, see
    // https://github.com/phetsims/scenery/issues/464 for mouse handling.
    if ( !( this.passiveEvents === true ) && ( callback !== this.mouseDown || platform.ie || platform.edge ) ) {
      // We cannot prevent a passive event, so don't try
      domEvent.preventDefault();
    }

    sceneryLog && sceneryLog.InputEvent && sceneryLog.pop();
  }

  /**
   * Fires all of our events that were batched into the batchedEvents array.
   * @public (scenery-internal)
   */
  fireBatchedEvents() {
    sceneryLog && sceneryLog.InputEvent && this.currentlyFiringEvents && sceneryLog.InputEvent(
      'REENTRANCE DETECTED' );
    // Don't re-entrantly enter our loop, see https://github.com/phetsims/balloons-and-static-electricity/issues/406
    if ( !this.currentlyFiringEvents && this.batchedEvents.length ) {
      sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent( 'Input.fireBatchedEvents length:' + this.batchedEvents.length );
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
   * Clears any batched events that we don't want to process.
   * @public (scenery-internal)
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
   * enter/exit events.
   * @public (scenery-internal)
   */
  validatePointers() {
    this.validatePointersAction.execute();
  }

  /**
   * Removes all non-Mouse pointers from internal tracking.
   * @public (scenery-internal)
   */
  removeTemporaryPointers() {
    const fakeDomEvent = {
      // TODO: Does this break anything
    };

    for ( let i = this.pointers.length - 1; i >= 0; i-- ) {
      const pointer = this.pointers[ i ];
      if ( !( pointer instanceof Mouse ) ) {
        this.pointers.splice( i, 1 );

        // Send exit events. As we can't get a DOM event, we'll send a fake object instead.
        //TODO: consider exit() not taking an event?
        const exitTrail = pointer.trail || new Trail( this.rootNode );
        this.exitEvents( pointer, fakeDomEvent, exitTrail, 0, true );
      }
    }
  }

  /**
   * Hooks up DOM listeners to whatever type of object we are going to listen to.
   * @public (scenery-internal)
   */
  connectListeners() {
    BrowserEvents.addDisplay( this.display, this.attachToWindow, this.passiveEvents );
  }

  /**
   * Removes DOM listeners from whatever type of object we were listening to.
   * @public (scenery-internal)
   */
  disconnectListeners() {
    BrowserEvents.removeDisplay( this.display, this.attachToWindow, this.passiveEvents );
  }

  /**
   * Extract a {Vector2} global coordinate point from an arbitrary DOM event.
   * @public (scenery-internal)
   *
   * @param {Event} domEvent
   * @returns {Vector2}
   */
  pointFromEvent( domEvent ) {
    const position = Vector2.createFromPool( domEvent.clientX, domEvent.clientY );
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
   * @private
   *
   * @param {Pointer} pointer
   */
  addPointer( pointer ) {
    this.pointers.push( pointer );

    // Callback for showing pointer events.  Optimized for performance.
    if ( this.pointerAddedListeners.length ) {
      for ( let i = 0; i < this.pointerAddedListeners.length; i++ ) {
        this.pointerAddedListeners[ i ]( pointer );
      }
    }
  }

  /**
   * Add a listener to be called when a Pointer is added.
   * TODO: Just use an emitter
   * @public
   * @param {function} listener
   */
  addPointerAddedListener( listener ) {
    this.pointerAddedListeners.push( listener );
  }

  /**
   * Remove a listener being called when a Pointer is added.
   * @public
   * @param listener
   */
  removePointerAddedListener( listener ) {
    const index = this.pointerAddedListeners.indexOf( listener );
    if ( index !== -1 ) {
      this.pointerAddedListeners.splice( index, index + 1 );
    }
  }

  /**
   * Removes a pointer from our list. If we get future events for it (based on the ID) it will be ignored.
   * @private
   *
   * @param {Pointer} pointer
   */
  removePointer( pointer ) {
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
   * @private
   *
   * NOTE: There are some cases where we may have prematurely "removed" a pointer.
   *
   * @param {number} id
   * @returns {Pointer|null}
   */
  findPointerById( id ) {
    let i = this.pointers.length;
    while ( i-- ) {
      const pointer = this.pointers[ i ];
      if ( pointer.id === id ) {
        return pointer;
      }
    }
    return null;
  }

  /**
   * Initializes the Mouse object on the first mouse event (this may never happen on touch devices).
   * @private
   */
  initMouse() {
    this.mouse = new Mouse();
    this.addPointer( this.mouse );
  }

  /**
   * Initializes the accessible pointer object on the first a11y event.
   * @private
   */
  initA11yPointer() {
    this.a11yPointer = new PDOMPointer( this.display );

    this.addPointer( this.a11yPointer );
  }

  /**
   * Steps to dispatch an a11y related event. Before dispatch, the PDOMPointer is initialized if it
   * hasn't been created yet and a userGestureEmitter emits to indicate that a user has begun an interaction.
   * @private
   *
   * @param {Trail} trail
   * @param {string} eventType
   * @param {Event} domEvent
   * @param {boolean} bubbles
   */
  dispatchA11yEvent( trail, eventType, domEvent, bubbles ) {
    Display.userGestureEmitter.emit();

    // This workaround hopefully won't be here forever, see ParallelDOM.setExcludeLabelSiblingFromInput() and https://github.com/phetsims/a11y-research/issues/156
    if ( !( domEvent.target && domEvent.target.hasAttribute( PDOMUtils.DATA_EXCLUDE_FROM_INPUT ) ) ) {

      // To support PhET-iO playback, any potential playback events downstream of this playback event must be marked as
      // non playback events. This is to prevent the PhET-iO playback engine from repeating those events. This is here
      // because, unlike visual pointer events, alternative input from the PDOM is often called from code instead of
      // being "top-level" user input at the root of the event hierarchy. See https://github.com/phetsims/gravity-force-lab/issues/242
      Tandem.PHET_IO_ENABLED && phet.phetio.dataStream.pushNonPlaybackable();

      this.dispatchEvent( trail, eventType, this.a11yPointer, domEvent, bubbles );

      Tandem.PHET_IO_ENABLED && phet.phetio.dataStream.popNonPlaybackable();
    }
  }

  /**
   * Update the PDOMPointer with a new trail from a DOMEvent and return it. For multiple dispatches from a single
   * DOMEvent, this ensures that all will dispatch to the same Trail.
   * @private
   *
   * @param domEvent
   * @returns {Trail}
   */
  updateTrailForPDOMDispatch( domEvent ) {
    if ( !this.a11yPointer ) { this.initA11yPointer(); }
    return this.a11yPointer.updateTrail( this.getTrailId( domEvent ) );
  }

  /**
   * Get the trail ID of the node represented by a DOM element in the accessible PDOM.
   * @private
   *
   * @param  {Event} domEvent
   * @returns {string}
   */
  getTrailId( domEvent ) {
    assert && assert( this.display._accessible, 'Display must be accessible to get trail IDs from AccessiblePeers' );
    assert && assert( domEvent.target || domEvent[ TARGET_SUBSTITUTE_KEY ], 'need a way to get the target' );

    // could be serialized event for phet-io playbacks, see Input.serializeDOMEvent()
    if ( domEvent[ TARGET_SUBSTITUTE_KEY ] ) {
      assert && assert( domEvent[ TARGET_SUBSTITUTE_KEY ] instanceof Object );
      return domEvent[ TARGET_SUBSTITUTE_KEY ][ PDOMUtils.DATA_TRAIL_ID ];
    }
    else {
      assert && assert( domEvent.target instanceof window.Element );
      return domEvent.target.getAttribute( PDOMUtils.DATA_TRAIL_ID );
    }
  }

  /**
   * Triggers a logical mousedown event.
   * @public (scenery-internal)
   *
   * NOTE: This may also be called from the pointer event handler (pointerDown) or from things like fuzzing or
   * playback. The event may be "faked" for certain purposes.
   *
   * @param {number|null} id
   * @param {Vector2} point
   * @param {Event} event
   */
  mouseDown( id, point, event ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( 'mouseDown(' + id + ', ' + Input.debugText( point, event ) + ');' );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.mouseDownAction.execute( id, point, event );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical mouseup event.
   * @public (scenery-internal)
   *
   * NOTE: This may also be called from the pointer event handler (pointerUp) or from things like fuzzing or
   * playback. The event may be "faked" for certain purposes.
   *
   * @param {Vector2} point
   * @param {Event} event
   */
  mouseUp( point, event ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( 'mouseUp(' + Input.debugText( point, event ) + ');' );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.mouseUpAction.execute( point, event );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical mousemove event.
   * @public (scenery-internal)
   *
   * NOTE: This may also be called from the pointer event handler (pointerMove) or from things like fuzzing or
   * playback. The event may be "faked" for certain purposes.
   *
   * @param {Vector2} point
   * @param {Event} event
   */
  mouseMove( point, event ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( 'mouseMove(' + Input.debugText( point, event ) + ');' );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.mouseMoveAction.execute( point, event );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical mouseover event (this does NOT correspond to the Scenery event, since this is for the display)
   * @public (scenery-internal)
   *
   * @param {Vector2} point
   * @param {Event} event
   */
  mouseOver( point, event ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( 'mouseOver(' + Input.debugText( point, event ) + ');' );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.mouseOverAction.execute( point, event );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical mouseout event (this does NOT correspond to the Scenery event, since this is for the display)
   * @public (scenery-internal)
   *
   * @param {Vector2} point
   * @param {Event} event
   */
  mouseOut( point, event ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( 'mouseOut(' + Input.debugText( point, event ) + ');' );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.mouseOutAction.execute( point, event );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical mouse-wheel/scroll event.
   * @public (scenery-internal)
   *
   * @param {Event} event
   */
  wheel( event ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( 'wheel(' + Input.debugText( null, event ) + ');' );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.wheelScrollAction.execute( event );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical touchstart event. This is called for each touch point in a 'raw' event.
   * @public (scenery-internal)
   *
   * NOTE: This may also be called from the pointer event handler (pointerDown) or from things like fuzzing or
   * playback. The event may be "faked" for certain purposes.
   *
   * @param {number} id
   * @param {Vector2} point
   * @param {Event} event
   */
  touchStart( id, point, event ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( 'touchStart(\'' + id + '\',' + Input.debugText( point, event ) + ');' );
    sceneryLog && sceneryLog.Input && sceneryLog.push();


    // ignore this touch event if it came from the PDOM
    if ( !this.isTargetUnderPDOM( event.target ) ) {
      this.touchStartAction.execute( id, point, event );
    }

    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical touchend event. This is called for each touch point in a 'raw' event.
   * @public (scenery-internal)
   *
   * NOTE: This may also be called from the pointer event handler (pointerUp) or from things like fuzzing or
   * playback. The event may be "faked" for certain purposes.
   *
   * @param {number} id
   * @param {Vector2} point
   * @param {Event} event
   */
  touchEnd( id, point, event ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( 'touchEnd(\'' + id + '\',' + Input.debugText( point, event ) + ');' );
    sceneryLog && sceneryLog.Input && sceneryLog.push();

    // ignore this touch event if it came from the PDOM
    if ( !this.isTargetUnderPDOM( event.target ) ) {
      this.touchEndAction.execute( id, point, event );
    }

    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical touchmove event. This is called for each touch point in a 'raw' event.
   * @public (scenery-internal)
   *
   * NOTE: This may also be called from the pointer event handler (pointerMove) or from things like fuzzing or
   * playback. The event may be "faked" for certain purposes.
   *
   * @param {number} id
   * @param {Vector2} point
   * @param {Event} event
   */
  touchMove( id, point, event ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( 'touchMove(\'' + id + '\',' + Input.debugText( point, event ) + ');' );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.touchMoveAction.execute( id, point, event );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical touchcancel event. This is called for each touch point in a 'raw' event.
   * @public (scenery-internal)
   *
   * NOTE: This may also be called from the pointer event handler (pointerCancel) or from things like fuzzing or
   * playback. The event may be "faked" for certain purposes.
   *
   * @param {number} id
   * @param {Vector2} point
   * @param {Event} event
   */
  touchCancel( id, point, event ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( 'touchCancel(\'' + id + '\',' + Input.debugText( point, event ) + ');' );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.touchCancelAction.execute( id, point, event );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical penstart event (e.g. a stylus). This is called for each pen point in a 'raw' event.
   * @public (scenery-internal)
   *
   * NOTE: This may also be called from the pointer event handler (pointerDown) or from things like fuzzing or
   * playback. The event may be "faked" for certain purposes.
   *
   * @param {number} id
   * @param {Vector2} point
   * @param {Event} event
   */
  penStart( id, point, event ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( 'penStart(\'' + id + '\',' + Input.debugText( point, event ) + ');' );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.penStartAction.execute( id, point, event );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical penend event (e.g. a stylus). This is called for each pen point in a 'raw' event.
   * @public (scenery-internal)
   *
   * NOTE: This may also be called from the pointer event handler (pointerUp) or from things like fuzzing or
   * playback. The event may be "faked" for certain purposes.
   *
   * @param {number} id
   * @param {Vector2} point
   * @param {Event} event
   */
  penEnd( id, point, event ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( 'penEnd(\'' + id + '\',' + Input.debugText( point, event ) + ');' );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.penEndAction.execute( id, point, event );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical penmove event (e.g. a stylus). This is called for each pen point in a 'raw' event.
   * @public (scenery-internal)
   *
   * NOTE: This may also be called from the pointer event handler (pointerMove) or from things like fuzzing or
   * playback. The event may be "faked" for certain purposes.
   *
   * @param {number} id
   * @param {Vector2} point
   * @param {Event} event
   */
  penMove( id, point, event ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( 'penMove(\'' + id + '\',' + Input.debugText( point, event ) + ');' );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.penMoveAction.execute( id, point, event );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Triggers a logical pencancel event (e.g. a stylus). This is called for each pen point in a 'raw' event.
   * @public (scenery-internal)
   *
   * NOTE: This may also be called from the pointer event handler (pointerCancel) or from things like fuzzing or
   * playback. The event may be "faked" for certain purposes.
   *
   * @param {number} id
   * @param {Vector2} point
   * @param {Event} event
   */
  penCancel( id, point, event ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( 'penCancel(\'' + id + '\',' + Input.debugText( point, event ) + ');' );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.penCancelAction.execute( id, point, event );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Handles a pointerdown event, forwarding it to the proper logical event.
   * @public (scenery-internal)
   *
   * @param {number} id
   * @param {string} type
   * @param {Vector2} point
   * @param {Event} event
   */
  pointerDown( id, type, point, event ) {
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
          throw new Error( 'Unknown pointer type: ' + type );
        }
    }
  }

  /**
   * Handles a pointerup event, forwarding it to the proper logical event.
   * @public (scenery-internal)
   *
   * @param {number} id
   * @param {string} type
   * @param {Vector2} point
   * @param {Event} event
   */
  pointerUp( id, type, point, event ) {
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
          throw new Error( 'Unknown pointer type: ' + type );
        }
    }
  }

  /**
   * Handles a pointercancel event, forwarding it to the proper logical event.
   * @public (scenery-internal)
   *
   * @param {number} id
   * @param {string} type
   * @param {Vector2} point
   * @param {Event} event
   */
  pointerCancel( id, type, point, event ) {
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
          console.log( 'Unknown pointer type: ' + type );
        }
    }
  }

  /**
   * Handles a gotpointercapture event, forwarding it to the proper logical event.
   * @public (scenery-internal)
   *
   * @param {number} id
   * @param {string} type
   * @param {Vector2} point
   * @param {Event} event
   */
  gotPointerCapture( id, type, point, event ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `gotPointerCapture('${id}',${Input.debugText( null, event )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.gotPointerCaptureAction.execute( id, event );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Handles a lostpointercapture event, forwarding it to the proper logical event.
   * @public (scenery-internal)
   *
   * @param {number} id
   * @param {string} type
   * @param {Vector2} point
   * @param {Event} event
   */
  lostPointerCapture( id, type, point, event ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( `lostPointerCapture('${id}',${Input.debugText( null, event )});` );
    sceneryLog && sceneryLog.Input && sceneryLog.push();
    this.lostPointerCaptureAction.execute( id, event );
    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Handles a pointermove event, forwarding it to the proper logical event.
   * @public (scenery-internal)
   *
   * @param {number} id
   * @param {string} type
   * @param {Vector2} point
   * @param {Event} event
   */
  pointerMove( id, type, point, event ) {
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
          console.log( 'Unknown pointer type: ' + type );
        }
    }
  }

  /**
   * Handles a pointerover event, forwarding it to the proper logical event.
   * @public (scenery-internal)
   *
   * @param {number} id
   * @param {string} type
   * @param {Vector2} point
   * @param {Event} event
   */
  pointerOver( id, type, point, event ) {
    // TODO: accumulate mouse/touch info in the object if needed?
    // TODO: do we want to branch change on these types of events?
  }

  /**
   * Handles a pointerout event, forwarding it to the proper logical event.
   * @public (scenery-internal)
   *
   * @param {number} id
   * @param {string} type
   * @param {Vector2} point
   * @param {Event} event
   */
  pointerOut( id, type, point, event ) {
    // TODO: accumulate mouse/touch info in the object if needed?
    // TODO: do we want to branch change on these types of events?
  }

  /**
   * Handles a pointerenter event, forwarding it to the proper logical event.
   * @public (scenery-internal)
   *
   * @param {number} id
   * @param {string} type
   * @param {Vector2} point
   * @param {Event} event
   */
  pointerEnter( id, type, point, event ) {
    // TODO: accumulate mouse/touch info in the object if needed?
    // TODO: do we want to branch change on these types of events?
  }

  /**
   * Handles a pointerleave event, forwarding it to the proper logical event.
   * @public (scenery-internal)
   *
   * @param {number} id
   * @param {string} type
   * @param {Vector2} point
   * @param {Event} event
   */
  pointerLeave( id, type, point, event ) {
    // TODO: accumulate mouse/touch info in the object if needed?
    // TODO: do we want to branch change on these types of events?
  }

  /**
   * When we get an unknown pointer event type (allowed in the spec, see
   * https://developer.mozilla.org/en-US/docs/Web/API/PointerEvent/pointerType), we'll try to guess the pointer type
   * so that we can properly start/end the interaction. NOTE: this can happen for an 'up' where we received a
   * proper type for a 'down', so thus we need the detection.
   * @private
   *
   * @param {string} type
   * @param {number} id
   * @returns {string}
   */
  handleUnknownPointerType( type, id ) {
    if ( type !== '' ) {
      return type;
    }
    return ( this.mouse && this.mouse.id === id ) ? 'mouse' : 'touch';
  }

  /**
   * Given a pointer reference, hit test it and determine the Trail that the pointer is over.
   * @private
   *
   * @param {Pointer}
   * @returns {Trail}
   */
  getPointerTrail( pointer ) {
    return this.rootNode.trailUnderPointer( pointer ) || new Trail( this.rootNode );
  }

  /**
   * Called for each logical "up" event, for any pointer type.
   * @private
   *
   * @param {Pointer} pointer
   * @param {Event} event
   * @param {boolean} pointChanged
   */
  upEvent( pointer, event, pointChanged ) {

    // if the event target is within the PDOM the AT is sending a fake pointer event to the document - do not
    // dispatch this since the PDOM should only handle Input.A11Y_EVENT_TYPES, and all other pointer input should
    // go through the Display div. Otherwise, activation will be duplicated when we handle pointer and PDOM events
    if ( this.isTargetUnderPDOM( event.target ) ) {
      return;
    }

    sceneryLog && sceneryLog.Input && sceneryLog.Input( 'upEvent ' + pointer.toString() + ' changed:' + pointChanged );
    sceneryLog && sceneryLog.Input && sceneryLog.push();

    assert && assert( pointer instanceof Pointer );
    assert && assert( typeof pointChanged === 'boolean' );

    // We'll use this trail for the entire dispatch of this event.
    const eventTrail = this.branchChangeEvents( pointer, event, pointChanged );

    this.dispatchEvent( eventTrail, 'up', pointer, event, true );

    // touch pointers are transient, so fire exit/out to the trail afterwards
    if ( pointer instanceof Touch ) {
      this.exitEvents( pointer, event, eventTrail, 0, true );
    }

    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Called for each logical "down" event, for any pointer type.
   * @private
   *
   * @param {Pointer} pointer
   * @param {Event} event
   * @param {boolean} pointChanged
   */
  downEvent( pointer, event, pointChanged ) {

    // if the event target is within the PDOM the AT is sending a fake pointer event to the document - do not
    // dispatch this since the PDOM should only handle Input.A11Y_EVENT_TYPES, and all other pointer input should
    // go through the Display div. Otherwise, activation will be duplicated when we handle pointer and PDOM events
    if ( this.isTargetUnderPDOM( event.target ) ) {
      return;
    }

    sceneryLog && sceneryLog.Input && sceneryLog.Input( 'downEvent ' + pointer.toString() + ' changed:' + pointChanged );
    sceneryLog && sceneryLog.Input && sceneryLog.push();

    assert && assert( pointer instanceof Pointer );
    assert && assert( typeof pointChanged === 'boolean' );

    // We'll use this trail for the entire dispatch of this event.
    const eventTrail = this.branchChangeEvents( pointer, event, pointChanged );

    // pdom
    let focusableNode = null;
    const trailAccessible = !eventTrail.rootNode()._rendererSummary.isNotAccessible();

    // If any node in the trail has accessible content
    if ( trailAccessible ) {
      const inDisplay = this.display.bounds.containsPoint( pointer.point );
      if ( inDisplay ) {

        // An AT might have sent a down event outside of the display, if this happened we will not remove focus.
        // Starting with the leaf most node, search for the closest accessible ancestor from the node under the
        // pointer.
        for ( let i = eventTrail.nodes.length - 1; i >= 0; i-- ) {
          if ( eventTrail.nodes[ i ].focusable ) {
            focusableNode = eventTrail.nodes[ i ];
            break;
          }
        }

        // Remove keyboard focus, but store element that is receiving interaction in case we resume .
        this.display.pointerFocus = focusableNode;
        Display.focus = null;
      }
    }

    // dispatch after handling display focus in case immediate focusout interferes
    this.dispatchEvent( eventTrail, 'down', pointer, event, true );

    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Called for each logical "move" event, for any pointer type.
   * @private
   *
   * @param {Pointer} pointer
   * @param {Event} event
   */
  moveEvent( pointer, event ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( 'moveEvent ' + pointer.toString() );
    sceneryLog && sceneryLog.Input && sceneryLog.push();

    assert && assert( pointer instanceof Pointer );

    // Always treat move events as "point changed"
    this.branchChangeEvents( pointer, event, true );

    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Called for each logical "cancel" event, for any pointer type.
   * @private
   *
   * @param {Pointer} pointer
   * @param {Event} event
   * @param {boolean} pointChanged
   */
  cancelEvent( pointer, event, pointChanged ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input( 'cancelEvent ' + pointer.toString() + ' changed:' + pointChanged );
    sceneryLog && sceneryLog.Input && sceneryLog.push();

    assert && assert( pointer instanceof Pointer );
    assert && assert( typeof pointChanged === 'boolean' );

    // We'll use this trail for the entire dispatch of this event.
    const eventTrail = this.branchChangeEvents( pointer, event, pointChanged );

    this.dispatchEvent( eventTrail, 'cancel', pointer, event, true );

    // touch pointers are transient, so fire exit/out to the trail afterwards
    if ( pointer instanceof Touch ) {
      this.exitEvents( pointer, event, eventTrail, 0, true );
    }

    sceneryLog && sceneryLog.Input && sceneryLog.pop();
  }

  /**
   * Dispatches any necessary events that would result from the pointer's trail changing.
   * @private
   *
   * This will send the necessary exit/enter events (on subtrails that have diverged between before/after), the
   * out/over events, and if flagged a move event.
   *
   * @param {Pointer} pointer
   * @param {Event|null} event
   * @param {boolean} sendMove - Whether to send move events
   * @returns {Trail} - The current trail of the pointer
   */
  branchChangeEvents( pointer, event, sendMove ) {
    sceneryLog && sceneryLog.Input && sceneryLog.Input(
      'branchChangeEvents: ' + pointer.toString() + ' sendMove:' + sendMove );
    sceneryLog && sceneryLog.Input && sceneryLog.push();

    assert && assert( pointer instanceof Pointer );
    assert && assert( typeof sendMove === 'boolean' );

    const trail = this.getPointerTrail( pointer );
    const oldTrail = pointer.trail || new Trail( this.rootNode ); // TODO: consider a static trail reference

    const lastNodeChanged = oldTrail.lastNode() !== trail.lastNode();

    const branchIndex = Trail.branchIndex( trail, oldTrail );
    const isBranchChange = branchIndex !== trail.length || branchIndex !== oldTrail.length;
    isBranchChange && sceneryLog && sceneryLog.Input && sceneryLog.Input(
      'changed from ' + oldTrail.toString() + ' to ' + trail.toString() );

    // event order matches http://www.w3.org/TR/DOM-Level-3-Events/#events-mouseevent-event-order
    if ( sendMove ) {
      this.dispatchEvent( trail, 'move', pointer, event, true );
    }

    // we want to approximately mimic http://www.w3.org/TR/DOM-Level-3-Events/#events-mouseevent-event-order
    // TODO: if a node gets moved down 1 depth, it may see both an exit and enter?
    if ( isBranchChange ) {
      this.exitEvents( pointer, event, oldTrail, branchIndex, lastNodeChanged );
      this.enterEvents( pointer, event, trail, branchIndex, lastNodeChanged );
    }

    pointer.trail = trail;

    sceneryLog && sceneryLog.Input && sceneryLog.pop();
    return trail;
  }

  /**
   * Triggers 'enter' events along a trail change, and an 'over' event on the leaf.
   * @private
   *
   * For example, if we change from a trail [ a, b, c, d, e ] => [ a, b, x, y ], it will fire:
   *
   * - enter y
   * - enter x
   * - over y (bubbles)
   *
   * @param {Pointer} pointer
   * @param {Event|null} event
   * @param {Trail} trail - The "new" trail
   * @param {number} branchIndex - The first index where the old and new trails have a different node. We will notify
   *                               for this node and all "descendant" nodes in the relevant trail.
   * @param {boolean} lastNodeChanged - If the last node didn't change, we won't sent an over event.
   */
  enterEvents( pointer, event, trail, branchIndex, lastNodeChanged ) {
    if ( trail.length > branchIndex ) {
      for ( let newIndex = trail.length - 1; newIndex >= branchIndex; newIndex-- ) {
        // TODO: for performance, we should mutate a trail instead of returning a slice.
        this.dispatchEvent( trail.slice( 0, newIndex + 1 ), 'enter', pointer, event, false );
      }
    }

    if ( lastNodeChanged ) {
      this.dispatchEvent( trail, 'over', pointer, event, true );
    }
  }

  /**
   * Triggers 'exit' events along a trail change, and an 'out' event on the leaf.
   * @private
   *
   * For example, if we change from a trail [ a, b, c, d, e ] => [ a, b, x, y ], it will fire:
   *
   * - out e (bubbles)
   * - exit c
   * - exit d
   * - exit e
   *
   * @param {Pointer} pointer
   * @param {Event|null} event
   * @param {Trail} trail - The "old" trail
   * @param {number} branchIndex - The first index where the old and new trails have a different node. We will notify
   *                               for this node and all "descendant" nodes in the relevant trail.
   * @param {boolean} lastNodeChanged - If the last node didn't change, we won't sent an out event.
   */
  exitEvents( pointer, event, trail, branchIndex, lastNodeChanged ) {
    if ( lastNodeChanged ) {
      this.dispatchEvent( trail, 'out', pointer, event, true );
    }

    if ( trail.length > branchIndex ) {
      for ( let oldIndex = branchIndex; oldIndex < trail.length; oldIndex++ ) {
        // TODO: for performance, we should mutate a trail instead of returning a slice.
        this.dispatchEvent( trail.slice( 0, oldIndex + 1 ), 'exit', pointer, event, false );
      }
    }
  }

  /**
   * Dispatch to all nodes in the Trail, optionally bubbling down from the leaf to the root.
   * @private
   *
   * @param {Trail} trail
   * @param {string} type
   * @param {Pointer} pointer
   * @param {Event|null} event
   * @param {boolean} bubbles - If bubbles is false, the event is only dispatched to the leaf node of the trail.
   */
  dispatchEvent( trail, type, pointer, event, bubbles ) {
    sceneryLog && sceneryLog.EventDispatch && sceneryLog.EventDispatch(
      type + ' trail:' + trail.toString() + ' pointer:' + pointer.toString() + ' at ' + pointer.point.toString() );
    sceneryLog && sceneryLog.EventDispatch && sceneryLog.push();

    assert && assert( trail, 'Falsy trail for dispatchEvent' );

    // NOTE: event is not immutable, as its currentTarget changes
    const inputEvent = new SceneryEvent( trail, type, pointer, event );

    // first run through the pointer's listeners to see if one of them will handle the event
    this.dispatchToListeners( pointer, pointer.getListeners(), type, inputEvent );

    // if not yet handled, run through the trail in order to see if one of them will handle the event
    // at the base of the trail should be the scene node, so the scene will be notified last
    this.dispatchToTargets( trail, type, pointer, inputEvent, bubbles );

    // Notify input listeners on the Display
    this.dispatchToListeners( pointer, this.display.getInputListeners(), type, inputEvent );

    sceneryLog && sceneryLog.EventDispatch && sceneryLog.pop();
  }

  /**
   * Notifies an array of listeners with a specific event.
   * @private
   *
   * @param {Pointer} pointer
   * @param {Array.<Object>} listeners - Should be a defensive array copy already.
   * @param {string} type
   * @param {SceneryEvent} inputEvent
   */
  dispatchToListeners( pointer, listeners, type, inputEvent ) {
    assert && assert( inputEvent instanceof SceneryEvent );

    if ( inputEvent.handled ) {
      return;
    }

    const specificType = pointer.type + type; // e.g. mouseup, touchup

    for ( let i = 0; i < listeners.length; i++ ) {
      const listener = listeners[ i ];

      if ( !inputEvent.aborted && listener[ specificType ] ) {
        sceneryLog && sceneryLog.EventDispatch && sceneryLog.EventDispatch( specificType );
        sceneryLog && sceneryLog.EventDispatch && sceneryLog.push();

        listener[ specificType ]( inputEvent );

        sceneryLog && sceneryLog.EventDispatch && sceneryLog.pop();
      }

      if ( !inputEvent.aborted && listener[ type ] ) {
        sceneryLog && sceneryLog.EventDispatch && sceneryLog.EventDispatch( type );
        sceneryLog && sceneryLog.EventDispatch && sceneryLog.push();

        listener[ type ]( inputEvent );

        sceneryLog && sceneryLog.EventDispatch && sceneryLog.pop();
      }
    }
  }

  /**
   * Dispatch to all nodes in the Trail, optionally bubbling down from the leaf to the root.
   * @private
   *
   * @param {Trail} trail
   * @param {string} type
   * @param {Pointer} pointer
   * @param {SceneryEvent} inputEvent
   * @param {boolean} bubbles - If bubbles is false, the event is only dispatched to the leaf node of the trail.
   */
  dispatchToTargets( trail, type, pointer, inputEvent, bubbles ) {
    assert && assert( inputEvent instanceof SceneryEvent );

    if ( inputEvent.aborted || inputEvent.handled ) {
      return;
    }

    for ( let i = trail.getLastInputEnabledIndex(); i >= 0; bubbles ? i-- : i = -1 ) {
      const target = trail.nodes[ i ];
      if ( target.isDisposed ) {
        continue;
      }

      inputEvent.currentTarget = target;

      this.dispatchToListeners( pointer, target.getInputListeners(), type, inputEvent );

      // if the input event was aborted or handled, don't follow the trail down another level
      if ( inputEvent.aborted || inputEvent.handled ) {
        return;
      }
    }
  }

  /**
   * Returns true if the Display is accessible and the element is a descendant of the Display PDOM.
   * @private
   *
   * @param {HTMLElement} element
   * @returns {boolean}
   */
  isTargetUnderPDOM( element ) {
    return this.display._accessible && this.display.accessibleDOMElement.contains( element );
  }

  /**
   * A listener for the document body that will capture any keydown for a11y before focus is within the root of
   * the PDOM or handled by scenery. This is mostly useful for platform specific workarounds or signifying to the
   * Display that user interaction has begun. Otherwise, most a11y listeners should instead go through dispatchEvent.
   * @private
   *
   * @param {Event} event
   */
  handleDocumentKeydown( event ) {

    // If navigating in full screen mode, prevent a bug where focus gets lost if fullscreen mode was initiated
    // from an iframe by keeping focus in the display. getNext/getPreviousFocusable will return active element
    // if there are no more elements in that direction. See https://github.com/phetsims/scenery/issues/883
    if ( FullScreen.isFullScreen() && event.keyCode === KeyboardUtils.KEY_TAB ) {
      const rootElement = this.display.accessibleDOMElement;
      const nextElement = event.shiftKey ? PDOMUtils.getPreviousFocusable( rootElement ) :
                          PDOMUtils.getNextFocusable( rootElement );
      if ( nextElement === event.target ) {
        event.preventDefault();
      }
    }

    // if an accessible Node was being interacted with a mouse, or had focus when sim is made inactive, this Node
    // should receive focus upon resuming keyboard navigation
    if ( this.display.pointerFocus || this.display.activeNode ) {
      const active = this.display.pointerFocus || this.display.activeNode;

      // if there is a single accessible instance, we can restore focus
      if ( active.getAccessibleInstances().length === 1 &&
           active.focusable && // if the active element is focusable
           active.accessibleInstances[ 0 ].trail.isVisible() && // if the active element is visible
           event.keyCode === KeyboardUtils.KEY_TAB ) {
        event.preventDefault();
        active.focus();
        this.display.pointerFocus = null;
        this.display.activeNode = null;
      }
    }
  }

  /**
   * Saves the main information we care about from a DOM `Event` into a JSON-like structure.
   * @public
   *
   * @param {Event} domEvent
   * @returns {Object} - see domEventPropertiesToSerialize for list keys that are serialized
   */
  static serializeDomEvent( domEvent ) {
    const entries = {};
    for ( const property in domEvent ) {

      // we shouldn't check if domEvent.hasOwnProperty because some properties come from supertypes
      if ( domEventPropertiesToSerialize[ property ] ) {

        const domEventProperty = domEvent[ property ];

        // stringifying dom event object properties can cause circular references, so we avoid that completely
        if ( property === 'touches' || property === 'targetTouches' || property === 'changedTouches' ) {

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

        // If the target came from the accessibility PDOM, then we want to store the Node trail id of where it came from.
        if ( property === 'target' && domEventProperty !== null &&

             // Some event targets like HTMLDocument may not have this method, see https://github.com/phetsims/scenery/issues/979
             typeof domEventProperty.getAttribute === 'function' &&

             // If false, then this target isn't a PDOM element, so we can skip this serialization
             domEventProperty.hasAttribute( PDOMUtils.DATA_TRAIL_ID ) ) {

          entries[ property ] = {};
          entries[ property ][ PDOMUtils.DATA_TRAIL_ID ] = domEventProperty.getAttribute( PDOMUtils.DATA_TRAIL_ID );
        }
        else {
          entries[ property ] = ( ( typeof domEventProperty === 'object' ) && ( domEventProperty !== null ) ? {} : JSON.parse( JSON.stringify( domEventProperty ) ) ); // TODO: is parse/stringify necessary?
        }
      }
    }
    return entries;
  }

  /**
   * From a serialized dom event, return a recreated window.Event
   * @public (scenery-internal)
   *
   * @param {Object} eventObject
   * @returns {Event}
   */
  static deserializeDomEvent( eventObject ) {

    // IE doesn't support `new Event()`, but document.createEvent is largely deprecated, so only use it when we must.
    const domEvent = platform.ie ? document.createEvent( 'Event' ) : new window.Event( 'Event' );
    for ( const key in eventObject ) {

      // `type` is readonly, so don't try to set it.
      if ( eventObject.hasOwnProperty( key ) && key !== 'type' ) {

        // Special case for target since we can't set that read-only property. Instead use a substitute key.
        if ( key === 'target' ) {
          domEvent[ TARGET_SUBSTITUTE_KEY ] = eventObject[ key ];
        }
        else {
          domEvent[ key ] = eventObject[ key ];
        }
      }
    }
    return domEvent;
  }

  /**
   * Convenience function for logging out a point/event combination.
   * @private
   *
   * @param {Vector2|null} point - Not logged if null
   * @param {Event} domEvent
   */
  static debugText( point, domEvent ) {
    let result = domEvent.timeStamp + ' ' + domEvent.type;
    if ( point !== null ) {
      result = point.x + ',' + point.y + ' ' + result;
    }
    return result;
  }

  /**
   * Maps the current MS pointer types onto the pointer spec.
   * @public (scenery-internal)
   *
   * @param {Event} event
   * @returns {string}
   */
  static msPointerType( event ) {
    if ( event.pointerType === window.MSPointerEvent.MSPOINTER_TYPE_TOUCH ) {
      return 'touch';
    }
    else if ( event.pointerType === window.MSPointerEvent.MSPOINTER_TYPE_PEN ) {
      return 'pen';
    }
    else if ( event.pointerType === window.MSPointerEvent.MSPOINTER_TYPE_MOUSE ) {
      return 'mouse';
    }
    else {
      return event.pointerType; // hope for the best
    }
  }
}

// @public {Array.<string>} - Basic event listener types that are not pointer-type specific
Input.BASIC_EVENT_TYPES = () => {
  return [ 'down', 'up', 'cancel', 'move', 'wheel', 'enter', 'exit', 'over', 'out' ];
};

// @public {Array.<string>} - Valid accessibility event types, these largely follow the HTML spec for the same names.
//                            See the doc at the top of this file for more information.
Input.A11Y_EVENT_TYPES = () => {
  return [ 'focus', 'blur', 'click', 'input', 'change', 'keydown', 'keyup', 'focusin', 'focusout' ];
};

// @public {Array.<string>} - Valid prefixes for the basic event types above
Input.EVENT_PREFIXES = () => {
  return [ '', 'mouse', 'touch', 'pen' ];
};

// @public {Array.<string>} - Includes basic and specific types, e.g. both 'up' and 'mouseup'
Input.ALL_EVENT_TYPES = () => {
  return Input.EVENT_PREFIXES.map( prefix => {
    return Input.BASIC_EVENT_TYPES.map( eventName => {
      return prefix + eventName;
    } );
  } ).concat( [ Input.A11Y_EVENT_TYPES ] );
};

scenery.register( 'Input', Input );
export default Input;