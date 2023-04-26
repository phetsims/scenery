// Copyright 2017-2023, University of Colorado Boulder

/**
 * Handles attaching/detaching and forwarding browser input events to displays.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import arrayRemove from '../../../phet-core/js/arrayRemove.js';
import platform from '../../../phet-core/js/platform.js';
import { BatchedDOMEventType, Display, EventContext, Features, FocusManager, globalKeyStateTracker, PDOMUtils, scenery } from '../imports.js';

// Sometimes we need to add a listener that does absolutely nothing
const noop = () => {};

// Ensure we only attach global window listeners (display independent) once
let isGloballyAttached = false;

const BrowserEvents = {

  // Prevents focus related event callbacks from being dispatched - scenery internal operations might change
  // focus temporarily, we don't want event listeners to be called in this case because they are transient and not
  // caused by user interaction.
  blockFocusCallbacks: false,

  /**
   * Adds a Display to the list of displays that will be notified of input events.
   * @public
   *
   * @param {Display} display
   * @param {boolean} attachToWindow - Whether events should be attached to the window. If false, they will be
   *                                   attached to the Display's domElement.
   * @param {boolean|null} passiveEvents - The value of the `passive` option for adding/removing DOM event listeners
   */
  addDisplay( display, attachToWindow, passiveEvents ) {
    assert && assert( display instanceof Display );
    assert && assert( typeof attachToWindow === 'boolean' );
    assert && assert( !_.includes( this.attachedDisplays, display ),
      'A display cannot be concurrently attached to events more than one time' );

    // Always first please
    if ( !isGloballyAttached ) {
      isGloballyAttached = true;

      // never unattach because we don't know if there are other Displays listening to this.
      globalKeyStateTracker.attachToWindow();
      FocusManager.attachToWindow();
    }

    this.attachedDisplays.push( display );

    if ( attachToWindow ) {
      // lazily connect listeners
      if ( this.attachedDisplays.length === 1 ) {
        this.connectWindowListeners( passiveEvents );
      }
    }
    else {
      this.addOrRemoveListeners( display.domElement, true, passiveEvents );
    }

    // Only add the wheel listeners directly on the elements, so it won't trigger outside
    display.domElement.addEventListener( 'wheel', this.onwheel, BrowserEvents.getEventOptions( passiveEvents, true ) );
  },

  /**
   * Removes a Display to the list of displays that will be notified of input events.
   * @public
   *
   * @param {Display} display
   * @param {boolean} attachToWindow - The value provided to addDisplay
   * @param {boolean|null} passiveEvents - The value of the `passive` option for adding/removing DOM event listeners
   */
  removeDisplay( display, attachToWindow, passiveEvents ) {
    assert && assert( display instanceof Display );
    assert && assert( typeof attachToWindow === 'boolean' );
    assert && assert( _.includes( this.attachedDisplays, display ),
      'This display was not already attached to listen for window events' );

    arrayRemove( this.attachedDisplays, display );

    // lazily disconnect listeners
    if ( attachToWindow ) {
      if ( this.attachedDisplays.length === 0 ) {
        this.disconnectWindowListeners( passiveEvents );
      }
    }
    else {
      this.addOrRemoveListeners( display.domElement, false, passiveEvents );
    }

    display.domElement.removeEventListener( 'wheel', this.onwheel, BrowserEvents.getEventOptions( passiveEvents, true ) );
  },

  /**
   * Returns the value to provide as the 3rd parameter to addEventListener/removeEventListener.
   * @private
   *
   * @param {boolean|null} passiveEvents
   * @param {boolean} isMain - If false, it is used on the "document" for workarounds.
   * @returns {Object|boolean}
   */
  getEventOptions( passiveEvents, isMain ) {
    const passDirectPassiveFlag = Features.passive && passiveEvents !== null;
    if ( !passDirectPassiveFlag ) {
      return false;
    }
    else {
      const eventOptions = { passive: passiveEvents };
      if ( isMain ) {
        eventOptions.capture = false;
      }

      assert && assert( !eventOptions.capture, 'Do not use capture without consulting globalKeyStateTracker, ' +
                                               'which expects have listeners called FIRST in keyboard-related cases.' );
      return eventOptions;
    }
  },

  /**
   * {number} - Will be checked/mutated when listeners are added/removed.
   * @private
   */
  listenersAttachedToWindow: 0,

  /**
   * {number} - Will be checked/mutated when listeners are added/removed.
   * @private
   */
  listenersAttachedToElement: 0,

  /**
   * {Array.<Display>} - All Displays that should have input events forwarded.
   * @private
   */
  attachedDisplays: [],

  /**
   * {boolean} - Whether pointer events in the format specified by the W3C specification are allowed.
   * @private
   *
   * NOTE: Pointer events are currently disabled for Firefox due to https://github.com/phetsims/scenery/issues/837.
   */
  canUsePointerEvents: !!( ( window.navigator && window.navigator.pointerEnabled ) || window.PointerEvent ) && !platform.firefox,

  /**
   * {boolean} - Whether pointer events in the format specified by the MS specification are allowed.
   * @private
   */
  canUseMSPointerEvents: window.navigator && window.navigator.msPointerEnabled,

  /**
   * {Array.<string>} - All W3C pointer event types that we care about.
   * @private
   */
  pointerListenerTypes: [
    'pointerdown',
    'pointerup',
    'pointermove',
    'pointerover',
    'pointerout',
    'pointercancel',
    'gotpointercapture',
    'lostpointercapture'
  ],

  /**
   * {Array.<string>} - All MS pointer event types that we care about.
   * @private
   */
  msPointerListenerTypes: [
    'MSPointerDown',
    'MSPointerUp',
    'MSPointerMove',
    'MSPointerOver',
    'MSPointerOut',
    'MSPointerCancel'
  ],

  /**
   * {Array.<string>} - All touch event types that we care about
   * @private
   */
  touchListenerTypes: [
    'touchstart',
    'touchend',
    'touchmove',
    'touchcancel'
  ],

  /**
   * {Array.<string>} - All mouse event types that we care about
   * @private
   */
  mouseListenerTypes: [
    'mousedown',
    'mouseup',
    'mousemove',
    'mouseover',
    'mouseout'
  ],

  /**
   * {Array.<string>} - All wheel event types that we care about
   * @private
   */
  wheelListenerTypes: [
    'wheel'
  ],

  /**
   * {Array.<string>} - Alternative input types
   * @private
   */
  altListenerTypes: PDOMUtils.DOM_EVENTS,

  /**
   * Returns all event types that will be listened to on this specific platform.
   * @private
   *
   * @returns {Array.<string>}
   */
  getNonWheelUsedTypes() {
    let eventTypes;

    if ( this.canUsePointerEvents ) {
      // accepts pointer events corresponding to the spec at http://www.w3.org/TR/pointerevents/
      sceneryLog && sceneryLog.Input && sceneryLog.Input( 'Detected pointer events support, using that instead of mouse/touch events' );

      eventTypes = this.pointerListenerTypes;
    }
    else if ( this.canUseMSPointerEvents ) {
      sceneryLog && sceneryLog.Input && sceneryLog.Input( 'Detected MS pointer events support, using that instead of mouse/touch events' );

      eventTypes = this.msPointerListenerTypes;
    }
    else {
      sceneryLog && sceneryLog.Input && sceneryLog.Input( 'No pointer events support detected, using mouse/touch events' );

      eventTypes = this.touchListenerTypes.concat( this.mouseListenerTypes );
    }

    eventTypes = eventTypes.concat( this.altListenerTypes );

    // eventTypes = eventTypes.concat( this.wheelListenerTypes );

    return eventTypes;
  },

  /**
   * Connects event listeners directly to the window.
   * @private
   *
   * @param {boolean|null} passiveEvents - The value of the `passive` option for adding/removing DOM event listeners
   */
  connectWindowListeners( passiveEvents ) {
    this.addOrRemoveListeners( window, true, passiveEvents );
  },

  /**
   * Disconnects event listeners from the window.
   * @private
   *
   * @param {boolean|null} passiveEvents - The value of the `passive` option for adding/removing DOM event listeners
   */
  disconnectWindowListeners( passiveEvents ) {
    this.addOrRemoveListeners( window, false, passiveEvents );
  },

  /**
   * Either adds or removes event listeners to an object, depending on the flag.
   * @private
   *
   * @param {*} element - The element (window or DOM element) to add listeners to.
   * @param {boolean} addOrRemove - If true, listeners will be added. If false, listeners will be removed.
   * @param {boolean|null} passiveEvents - The value of the `passive` option for adding/removing DOM event listeners
   *                                       NOTE: if it is passed in as null, the default value for the browser will be
   *                                       used.
   */
  addOrRemoveListeners( element, addOrRemove, passiveEvents ) {
    assert && assert( typeof addOrRemove === 'boolean' );
    assert && assert( typeof passiveEvents === 'boolean' || passiveEvents === null );

    const forWindow = element === window;
    assert && assert( !forWindow || ( this.listenersAttachedToWindow > 0 ) === !addOrRemove,
      'Do not add listeners to the window when already attached, or remove listeners when none are attached' );

    const delta = addOrRemove ? 1 : -1;
    if ( forWindow ) {
      this.listenersAttachedToWindow += delta;
    }
    else {
      this.listenersAttachedToElement += delta;
    }
    assert && assert( this.listenersAttachedToWindow === 0 || this.listenersAttachedToElement === 0,
      'Listeners should not be added both with addDisplayToWindow and addDisplayToElement. Use only one.' );

    const method = addOrRemove ? 'addEventListener' : 'removeEventListener';

    // {Array.<string>}
    const eventTypes = this.getNonWheelUsedTypes();

    for ( let i = 0; i < eventTypes.length; i++ ) {
      const type = eventTypes[ i ];

      // If we add input listeners to the window itself, iOS Safari 7 won't send touch events to displays in an
      // iframe unless we also add dummy listeners to the document.
      if ( forWindow ) {
        // Workaround for older browsers needed,
        // see https://developer.mozilla.org/en-US/docs/Web/API/EventTarget/addEventListener#Improving_scrolling_performance_with_passive_listeners
        document[ method ]( type, noop, BrowserEvents.getEventOptions( passiveEvents, false ) );
      }

      const callback = this[ `on${type}` ];
      assert && assert( !!callback );

      // Workaround for older browsers needed,
      // see https://developer.mozilla.org/en-US/docs/Web/API/EventTarget/addEventListener#Improving_scrolling_performance_with_passive_listeners
      element[ method ]( type, callback, BrowserEvents.getEventOptions( passiveEvents, true ) );
    }
  },

  /**
   * Sets an event from the window to be batched on all of the displays.
   * @private
   *
   * @param {EventContext} eventContext
   * @param {BatchedDOMEventType} batchType - TODO: turn to full enumeration?
   * @param {string} inputCallbackName - e.g. 'mouseDown', will trigger Input.mouseDown
   * @param {boolean} triggerImmediate - Whether this will be force-executed now, causing all batched events to fire.
   *                                     Useful for events (like mouseup) that responding synchronously is
   *                                     necessary for certain security-sensitive actions (like triggering
   *                                     full-screen).
   */
  batchWindowEvent( eventContext, batchType, inputCallbackName, triggerImmediate ) {
    // NOTE: For now, we don't check whether the event is actually within the display's boundingClientRect. Most
    // displays will want to receive events outside of their bounds (especially for checking drags and mouse-ups
    // outside of their bounds).
    for ( let i = 0; i < this.attachedDisplays.length; i++ ) {
      const display = this.attachedDisplays[ i ];
      const input = display._input;

      if ( !BrowserEvents.blockFocusCallbacks || ( inputCallbackName !== 'focusIn' && inputCallbackName !== 'focusOut' ) ) {
        input.batchEvent( eventContext, batchType, input[ inputCallbackName ], triggerImmediate );
      }
    }
  },

  /**
   * Listener for window's pointerdown event.
   * @private
   *
   * @param {Event} domEvent
   */
  onpointerdown: function onpointerdown( domEvent ) {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'pointerdown' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // Get the active element BEFORE any actions are taken
    const eventContext = new EventContext( domEvent );

    if ( domEvent.pointerType === 'mouse' ) {
      Display.userGestureEmitter.emit();
    }

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( eventContext, BatchedDOMEventType.POINTER_TYPE, 'pointerDown', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  },

  /**
   * Listener for window's pointerup event.
   * @private
   *
   * @param {Event} domEvent
   */
  onpointerup: function onpointerup( domEvent ) {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'pointerup' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // Get the active element BEFORE any actions are taken
    const eventContext = new EventContext( domEvent );

    Display.userGestureEmitter.emit();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( eventContext, BatchedDOMEventType.POINTER_TYPE, 'pointerUp', true );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  },

  /**
   * Listener for window's pointermove event.
   * @private
   *
   * @param {Event} domEvent
   */
  onpointermove: function onpointermove( domEvent ) {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'pointermove' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.POINTER_TYPE, 'pointerMove', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  },

  /**
   * Listener for window's pointerover event.
   * @private
   *
   * @param {Event} domEvent
   */
  onpointerover: function onpointerover( domEvent ) {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'pointerover' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.POINTER_TYPE, 'pointerOver', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  },

  /**
   * Listener for window's pointerout event.
   * @private
   *
   * @param {Event} domEvent
   */
  onpointerout: function onpointerout( domEvent ) {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'pointerout' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.POINTER_TYPE, 'pointerOut', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  },

  /**
   * Listener for window's pointercancel event.
   * @private
   *
   * @param {Event} domEvent
   */
  onpointercancel: function onpointercancel( domEvent ) {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'pointercancel' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.POINTER_TYPE, 'pointerCancel', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  },

  /**
   * Listener for window's gotpointercapture event.
   * @private
   *
   * @param {Event} domEvent
   */
  ongotpointercapture: function ongotpointercapture( domEvent ) {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'gotpointercapture' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.POINTER_TYPE, 'gotPointerCapture', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  },

  /**
   * Listener for window's lostpointercapture event.
   * @private
   *
   * @param {Event} domEvent
   */
  onlostpointercapture: function onlostpointercapture( domEvent ) {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'lostpointercapture' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.POINTER_TYPE, 'lostPointerCapture', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  },

  /**
   * Listener for window's MSPointerDown event.
   * @private
   *
   * @param {Event} domEvent
   */
  onMSPointerDown: function onMSPointerDown( domEvent ) {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'MSPointerDown' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.MS_POINTER_TYPE, 'pointerDown', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  },

  /**
   * Listener for window's MSPointerUp event.
   * @private
   *
   * @param {Event} domEvent
   */
  onMSPointerUp: function onMSPointerUp( domEvent ) {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'MSPointerUp' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.MS_POINTER_TYPE, 'pointerUp', true );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  },

  /**
   * Listener for window's MSPointerMove event.
   * @private
   *
   * @param {Event} domEvent
   */
  onMSPointerMove: function onMSPointerMove( domEvent ) {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'MSPointerMove' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.MS_POINTER_TYPE, 'pointerMove', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  },

  /**
   * Listener for window's MSPointerOver event.
   * @private
   *
   * @param {Event} domEvent
   */
  onMSPointerOver: function onMSPointerOver( domEvent ) {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'MSPointerOver' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.MS_POINTER_TYPE, 'pointerOver', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  },

  /**
   * Listener for window's MSPointerOut event.
   * @private
   *
   * @param {Event} domEvent
   */
  onMSPointerOut: function onMSPointerOut( domEvent ) {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'MSPointerOut' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.MS_POINTER_TYPE, 'pointerOut', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  },

  /**
   * Listener for window's MSPointerCancel event.
   * @private
   *
   * @param {Event} domEvent
   */
  onMSPointerCancel: function onMSPointerCancel( domEvent ) {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'MSPointerCancel' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.MS_POINTER_TYPE, 'pointerCancel', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  },

  /**
   * Listener for window's touchstart event.
   * @private
   *
   * @param {Event} domEvent
   */
  ontouchstart: function ontouchstart( domEvent ) {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'touchstart' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.TOUCH_TYPE, 'touchStart', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  },

  /**
   * Listener for window's touchend event.
   * @private
   *
   * @param {Event} domEvent
   */
  ontouchend: function ontouchend( domEvent ) {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'touchend' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // Get the active element BEFORE any actions are taken
    const eventContext = new EventContext( domEvent );

    Display.userGestureEmitter.emit();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( eventContext, BatchedDOMEventType.TOUCH_TYPE, 'touchEnd', true );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  },

  /**
   * Listener for window's touchmove event.
   * @private
   *
   * @param {Event} domEvent
   */
  ontouchmove: function ontouchmove( domEvent ) {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'touchmove' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.TOUCH_TYPE, 'touchMove', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  },

  /**
   * Listener for window's touchcancel event.
   * @private
   *
   * @param {Event} domEvent
   */
  ontouchcancel: function ontouchcancel( domEvent ) {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'touchcancel' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.TOUCH_TYPE, 'touchCancel', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  },

  /**
   * Listener for window's mousedown event.
   * @private
   *
   * @param {Event} domEvent
   */
  onmousedown: function onmousedown( domEvent ) {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'mousedown' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // Get the active element BEFORE any actions are taken
    const eventContext = new EventContext( domEvent );

    Display.userGestureEmitter.emit();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( eventContext, BatchedDOMEventType.MOUSE_TYPE, 'mouseDown', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  },

  /**
   * Listener for window's mouseup event.
   * @private
   *
   * @param {Event} domEvent
   */
  onmouseup: function onmouseup( domEvent ) {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'mouseup' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // Get the active element BEFORE any actions are taken
    const eventContext = new EventContext( domEvent );

    Display.userGestureEmitter.emit();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( eventContext, BatchedDOMEventType.MOUSE_TYPE, 'mouseUp', true );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  },

  /**
   * Listener for window's mousemove event.
   * @private
   *
   * @param {Event} domEvent
   */
  onmousemove: function onmousemove( domEvent ) {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'mousemove' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.MOUSE_TYPE, 'mouseMove', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  },

  /**
   * Listener for window's mouseover event.
   * @private
   *
   * @param {Event} domEvent
   */
  onmouseover: function onmouseover( domEvent ) {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'mouseover' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.MOUSE_TYPE, 'mouseOver', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  },

  /**
   * Listener for window's mouseout event.
   * @private
   *
   * @param {Event} domEvent
   */
  onmouseout: function onmouseout( domEvent ) {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'mouseout' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.MOUSE_TYPE, 'mouseOut', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  },

  /**
   * Listener for window's wheel event.
   * @private
   *
   * @param {Event} domEvent
   */
  onwheel: function onwheel( domEvent ) {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'wheel' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.WHEEL_TYPE, 'wheel', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  },

  onfocusin: function onfocusin( domEvent ) {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'focusin' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.

    // if ( domEvent.target.id === 'display1-primary-30-44-2795-2802-2797-2806-3012-3011-2992' ) {
    //   debugger;
    // }

    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.ALT_TYPE, 'focusIn', true );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  },

  onfocusout: function onfocusout( domEvent ) {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'focusout' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.ALT_TYPE, 'focusOut', true );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  },

  oninput: function oninput( domEvent ) {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'input' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.

    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.ALT_TYPE, 'input', true );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  },

  onchange: function onchange( domEvent ) {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'change' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.ALT_TYPE, 'change', true );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  },

  onclick: function onclick( domEvent ) {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'click' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.ALT_TYPE, 'click', true );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  },

  onkeydown: function onkeydown( domEvent ) {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'keydown' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.ALT_TYPE, 'keyDown', true );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  },

  onkeyup: function onkeyup( domEvent ) {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'keyup' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.ALT_TYPE, 'keyUp', true );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  }
};

scenery.register( 'BrowserEvents', BrowserEvents );

export default BrowserEvents;