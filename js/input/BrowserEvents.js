// Copyright 2013-2016, University of Colorado Boulder

/**
 * Handles attaching/detaching and forwarding browser input events to displays.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var arrayRemove = require( 'PHET_CORE/arrayRemove' );
  var Features = require( 'SCENERY/util/Features' );
  var scenery = require( 'SCENERY/scenery' );
  var BatchedDOMEvent = require( 'SCENERY/input/BatchedDOMEvent' );

  // Sometimes we need to add a listener that does absolutely nothing
  var noop = function noop() {};

  var BrowserEvents = {
    /**
     * Adds a Display to the list of displays that will be notified of input events.
     * @public
     *
     * @param {Display} display
     * @param {boolean} attachToWindow - Whether events should be attached to the window. If false, they will be
     *                                   attached to the Display's domElement.
     */
    addDisplay: function( display, attachToWindow, passiveEvents ) {
      assert && assert( display instanceof scenery.Display );
      assert && assert( typeof attachToWindow === 'boolean' );
      assert && assert( !_.includes( this.attachedDisplays, display ),
        'A display cannot be concurrently attached to events more than one time' );

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
     */
    removeDisplay: function( display, attachToWindow, passiveEvents ) {
      assert && assert( display instanceof scenery.Display );
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

    getEventOptions: function( passiveEvents, isMain ) {
      var passDirectPassiveFlag = Features.passive && passiveEvents !== null;
      if ( !passDirectPassiveFlag ) {
        return false;
      }
      if ( isMain ) {
        return {
          useCapture: false,
          passive: passiveEvents
        };
      }
      else {
        return {
          passive: passiveEvents
        };
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
     */
    canUsePointerEvents: window.navigator && window.navigator.pointerEnabled,

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
     * Returns all event types that will be listened to on this specific platform.
     * @private
     *
     * @returns {Array.<string>}
     */
    getNonWheelUsedTypes: function() {
      var eventTypes;

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

      // eventTypes = eventTypes.concat( this.wheelListenerTypes );

      assert && assert( !_.includes( eventTypes, 'keydown' ),
        'Make sure not to preventDefault key events in the future.' );

      return eventTypes;
    },

    /**
     * Connects event listeners directly to the window.
     * @private
     */
    connectWindowListeners: function( passiveEvents ) {
      this.addOrRemoveListeners( window, true, passiveEvents );
    },

    /**
     * Disconnects event listeners from the window.
     * @private
     */
    disconnectWindowListeners: function( passiveEvents ) {
      this.addOrRemoveListeners( window, false, passiveEvents );
    },

    /**
     * Either adds or removes event listeners to an object, depending on the flag.
     * @private
     *
     * @param {*} element - The element (window or DOM element) to add listeners to.
     * @param {boolean} addOrRemove - If true, listeners will be added. If false, listeners will be removed.
     */
    addOrRemoveListeners: function( element, addOrRemove, passiveEvents ) {
      var documentOptions = BrowserEvents.getEventOptions( passiveEvents, false );
      var mainOptions = BrowserEvents.getEventOptions( passiveEvents, true );

      assert && assert( typeof addOrRemove === 'boolean' );

      var forWindow = element === window;
      assert && assert( !forWindow || ( this.listenersAttachedToWindow > 0 ) === !addOrRemove,
        'Do not add listeners to the window when already attached, or remove listeners when none are attached' );

      var delta = addOrRemove ? 1 : -1;
      if ( forWindow ) {
        this.listenersAttachedToWindow += delta;
      }
      else {
        this.listenersAttachedToElement += delta;
      }
      assert && assert( this.listenersAttachedToWindow === 0 || this.listenersAttachedToElement === 0,
        'Listeners should not be added both with addDisplayToWindow and addDisplayToElement. Use only one.' );

      var method = addOrRemove ? 'addEventListener' : 'removeEventListener';

      // {Array.<string>}
      var eventTypes = this.getNonWheelUsedTypes();

      for ( var i = 0; i < eventTypes.length; i++ ) {
        var type = eventTypes[ i ];

        // If we add input listeners to the window itself, iOS Safari 7 won't send touch events to displays in an
        // iframe unless we also add dummy listeners to the document.
        if ( forWindow ) {
          document[ method ]( type, noop, documentOptions );
        }

        var callback = this[ 'on' + type ];
        assert && assert( !!callback );

        element[ method ]( type, callback, mainOptions ); // false: don't use event capture for now
      }
    },

    /**
     * Sets an event from the window to be batched on all of the displays.
     * @private
     *
     * @param {Event} domEvent
     * @param {BatchedDOMEvent.Type} - TODO: turn to full enumeration?
     * @param {string} inputCallbackName - e.g. 'mouseDown', will trigger Input.mouseDown
     * @param {boolean} triggerImmediate - Whether this will be force-executed now, causing all batched events to fire.
     *                                     Useful for events (like mouseup) that responding synchronously is
     *                                     necessary for certain security-sensitive actions (like triggering
     *                                     full-screen).
     */
    batchWindowEvent: function( domEvent, batchType, inputCallbackName, triggerImmediate ) {
      // NOTE: For now, we don't check whether the event is actually within the display's boundingClientRect. Most
      // displays will want to receive events outside of their bounds (especially for checking drags and mouse-ups
      // outside of their bounds).
      for ( var i = 0; i < this.attachedDisplays.length; i++ ) {
        var display = this.attachedDisplays[ i ];
        var input = display._input;
        input.batchEvent( domEvent, batchType, input[ inputCallbackName ], triggerImmediate );
      }
    },

    /**
     * Listener for window's pointerdown event.
     * @private
     *
     * @param {Event} domEvent
     */
    onpointerdown: function onpointerdown( domEvent ) {
      if ( domEvent.pointerType === 'mouse' ) {
        scenery.Display.userGestureEmitter.emit();
      }
      // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
      BrowserEvents.batchWindowEvent( domEvent, BatchedDOMEvent.POINTER_TYPE, 'pointerDown', false );
    },

    /**
     * Listener for window's pointerup event.
     * @private
     *
     * @param {Event} domEvent
     */
    onpointerup: function onpointerup( domEvent ) {
      scenery.Display.userGestureEmitter.emit();
      // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
      BrowserEvents.batchWindowEvent( domEvent, BatchedDOMEvent.POINTER_TYPE, 'pointerUp', true );
    },

    /**
     * Listener for window's pointermove event.
     * @private
     *
     * @param {Event} domEvent
     */
    onpointermove: function onpointermove( domEvent ) {
      // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
      BrowserEvents.batchWindowEvent( domEvent, BatchedDOMEvent.POINTER_TYPE, 'pointerMove', false );
    },

    /**
     * Listener for window's pointerover event.
     * @private
     *
     * @param {Event} domEvent
     */
    onpointerover: function onpointerover( domEvent ) {
      // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
      BrowserEvents.batchWindowEvent( domEvent, BatchedDOMEvent.POINTER_TYPE, 'pointerOver', false );
    },

    /**
     * Listener for window's pointerout event.
     * @private
     *
     * @param {Event} domEvent
     */
    onpointerout: function onpointerout( domEvent ) {
      // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
      BrowserEvents.batchWindowEvent( domEvent, BatchedDOMEvent.POINTER_TYPE, 'pointerOut', false );
    },

    /**
     * Listener for window's pointercancel event.
     * @private
     *
     * @param {Event} domEvent
     */
    onpointercancel: function onpointercancel( domEvent ) {
      // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
      BrowserEvents.batchWindowEvent( domEvent, BatchedDOMEvent.POINTER_TYPE, 'pointerCancel', false );
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
      BrowserEvents.batchWindowEvent( domEvent, BatchedDOMEvent.POINTER_TYPE, 'gotPointerCapture', false );

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
      BrowserEvents.batchWindowEvent( domEvent, BatchedDOMEvent.POINTER_TYPE, 'lostPointerCapture', false );

      sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
    },

    /**
     * Listener for window's MSPointerDown event.
     * @private
     *
     * @param {Event} domEvent
     */
    onMSPointerDown: function onMSPointerDown( domEvent ) {
      // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
      BrowserEvents.batchWindowEvent( domEvent, BatchedDOMEvent.MS_POINTER_TYPE, 'pointerDown', false );
    },

    /**
     * Listener for window's MSPointerUp event.
     * @private
     *
     * @param {Event} domEvent
     */
    onMSPointerUp: function onMSPointerUp( domEvent ) {
      // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
      BrowserEvents.batchWindowEvent( domEvent, BatchedDOMEvent.MS_POINTER_TYPE, 'pointerUp', true );
    },

    /**
     * Listener for window's MSPointerMove event.
     * @private
     *
     * @param {Event} domEvent
     */
    onMSPointerMove: function onMSPointerMove( domEvent ) {
      // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
      BrowserEvents.batchWindowEvent( domEvent, BatchedDOMEvent.MS_POINTER_TYPE, 'pointerMove', false );
    },

    /**
     * Listener for window's MSPointerOver event.
     * @private
     *
     * @param {Event} domEvent
     */
    onMSPointerOver: function onMSPointerOver( domEvent ) {
      // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
      BrowserEvents.batchWindowEvent( domEvent, BatchedDOMEvent.MS_POINTER_TYPE, 'pointerOver', false );
    },

    /**
     * Listener for window's MSPointerOut event.
     * @private
     *
     * @param {Event} domEvent
     */
    onMSPointerOut: function onMSPointerOut( domEvent ) {
      // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
      BrowserEvents.batchWindowEvent( domEvent, BatchedDOMEvent.MS_POINTER_TYPE, 'pointerOut', false );
    },

    /**
     * Listener for window's MSPointerCancel event.
     * @private
     *
     * @param {Event} domEvent
     */
    onMSPointerCancel: function onMSPointerCancel( domEvent ) {
      // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
      BrowserEvents.batchWindowEvent( domEvent, BatchedDOMEvent.MS_POINTER_TYPE, 'pointerCancel', false );
    },

    /**
     * Listener for window's touchstart event.
     * @private
     *
     * @param {Event} domEvent
     */
    ontouchstart: function ontouchstart( domEvent ) {
      // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
      BrowserEvents.batchWindowEvent( domEvent, BatchedDOMEvent.TOUCH_TYPE, 'touchStart', false );
    },

    /**
     * Listener for window's touchend event.
     * @private
     *
     * @param {Event} domEvent
     */
    ontouchend: function ontouchend( domEvent ) {
      scenery.Display.userGestureEmitter.emit();
      // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
      BrowserEvents.batchWindowEvent( domEvent, BatchedDOMEvent.TOUCH_TYPE, 'touchEnd', true );
    },

    /**
     * Listener for window's touchmove event.
     * @private
     *
     * @param {Event} domEvent
     */
    ontouchmove: function ontouchmove( domEvent ) {
      // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
      BrowserEvents.batchWindowEvent( domEvent, BatchedDOMEvent.TOUCH_TYPE, 'touchMove', false );
    },

    /**
     * Listener for window's touchcancel event.
     * @private
     *
     * @param {Event} domEvent
     */
    ontouchcancel: function ontouchcancel( domEvent ) {
      // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
      BrowserEvents.batchWindowEvent( domEvent, BatchedDOMEvent.TOUCH_TYPE, 'touchCancel', false );
    },

    /**
     * Listener for window's mousedown event.
     * @private
     *
     * @param {Event} domEvent
     */
    onmousedown: function onmousedown( domEvent ) {
      scenery.Display.userGestureEmitter.emit();
      // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
      BrowserEvents.batchWindowEvent( domEvent, BatchedDOMEvent.MOUSE_TYPE, 'mouseDown', false );
    },

    /**
     * Listener for window's mouseup event.
     * @private
     *
     * @param {Event} domEvent
     */
    onmouseup: function onmouseup( domEvent ) {
      scenery.Display.userGestureEmitter.emit();
      // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
      BrowserEvents.batchWindowEvent( domEvent, BatchedDOMEvent.MOUSE_TYPE, 'mouseUp', true );
    },

    /**
     * Listener for window's mousemove event.
     * @private
     *
     * @param {Event} domEvent
     */
    onmousemove: function onmousemove( domEvent ) {
      // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
      BrowserEvents.batchWindowEvent( domEvent, BatchedDOMEvent.MOUSE_TYPE, 'mouseMove', false );
    },

    /**
     * Listener for window's mouseover event.
     * @private
     *
     * @param {Event} domEvent
     */
    onmouseover: function onmouseover( domEvent ) {
      // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
      BrowserEvents.batchWindowEvent( domEvent, BatchedDOMEvent.MOUSE_TYPE, 'mouseOver', false );
    },

    /**
     * Listener for window's mouseout event.
     * @private
     *
     * @param {Event} domEvent
     */
    onmouseout: function onmouseout( domEvent ) {
      // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
      BrowserEvents.batchWindowEvent( domEvent, BatchedDOMEvent.MOUSE_TYPE, 'mouseOut', false );
    },

    /**
     * Listener for window's wheel event.
     * @private
     *
     * @param {Event} domEvent
     */
    onwheel: function onwheel( domEvent ) {
      // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
      BrowserEvents.batchWindowEvent( domEvent, BatchedDOMEvent.WHEEL_TYPE, 'wheel', false );
    }
  };

  scenery.register( 'BrowserEvents', BrowserEvents );

  return BrowserEvents;
} );
