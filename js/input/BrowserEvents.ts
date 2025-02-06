// Copyright 2017-2025, University of Colorado Boulder

/**
 * Handles attaching/detaching and forwarding browser input events to displays.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import arrayRemove from '../../../phet-core/js/arrayRemove.js';
import FocusManager from '../accessibility/FocusManager.js';
import globalKeyStateTracker from '../accessibility/globalKeyStateTracker.js';
import PDOMUtils from '../accessibility/pdom/PDOMUtils.js';
import type Display from '../display/Display.js';
import { BatchedDOMEventType } from '../input/BatchedDOMEvent.js';
import EventContext from '../input/EventContext.js';
import scenery from '../scenery.js';
import Features from '../util/Features.js';
import DisplayGlobals from '../display/DisplayGlobals.js';

// Sometimes we need to add a listener that does absolutely nothing
const noop = () => {
  // no-op
};

// Ensure we only attach global window listeners (display independent) once
let isGloballyAttached = false;

export default class BrowserEvents {

  // Prevents focus related event callbacks from being dispatched - scenery internal operations might change
  // focus temporarily, we don't want event listeners to be called in this case because they are transient and not
  // caused by user interaction.
  public static blockFocusCallbacks = false;

  // True while Scenery is dispatching focus and blur related events. Scenery (PDOMTree) needs to restore focus
  // after operations, but that can be very buggy while focus events are already being handled.
  public static dispatchingFocusEvents = false;

  /**
   * Adds a Display to the list of displays that will be notified of input events.
   *
   * @param display
   * @param attachToWindow - Whether events should be attached to the window. If false, they will be
   *                                   attached to the Display's domElement.
   * @param passiveEvents - The value of the `passive` option for adding/removing DOM event listeners
   */
  public static addDisplay(
    display: Display,
    attachToWindow: boolean,
    passiveEvents: boolean | null
  ): void {
    assert && assert( !_.includes( BrowserEvents.attachedDisplays, display ),
      'A display cannot be concurrently attached to events more than one time' );

    // Always first please
    if ( !isGloballyAttached ) {
      isGloballyAttached = true;

      // never unattach because we don't know if there are other Displays listening to this.
      globalKeyStateTracker.attachToWindow();
      FocusManager.attachToWindow();
    }

    BrowserEvents.attachedDisplays.push( display );

    if ( attachToWindow ) {
      // lazily connect listeners
      if ( BrowserEvents.attachedDisplays.length === 1 ) {
        BrowserEvents.connectWindowListeners( passiveEvents );
      }
    }
    else {
      BrowserEvents.addOrRemoveListeners( display.domElement, true, passiveEvents );
    }

    // Only add the wheel listeners directly on the elements, so it won't trigger outside
    display.domElement.addEventListener( 'wheel', BrowserEvents.onwheel, BrowserEvents.getEventOptions( passiveEvents, true ) );
  }

  /**
   * Removes a Display to the list of displays that will be notified of input events.
   *
   * @param display
   * @param attachToWindow - The value provided to addDisplay
   * @param passiveEvents - The value of the `passive` option for adding/removing DOM event listeners
   */
  public static removeDisplay(
    display: Display,
    attachToWindow: boolean,
    passiveEvents: boolean | null
  ): void {
    assert && assert( _.includes( BrowserEvents.attachedDisplays, display ),
      'This display was not already attached to listen for window events' );

    arrayRemove( BrowserEvents.attachedDisplays, display );

    // lazily disconnect listeners
    if ( attachToWindow ) {
      if ( BrowserEvents.attachedDisplays.length === 0 ) {
        BrowserEvents.disconnectWindowListeners( passiveEvents );
      }
    }
    else {
      BrowserEvents.addOrRemoveListeners( display.domElement, false, passiveEvents );
    }

    display.domElement.removeEventListener( 'wheel', BrowserEvents.onwheel, BrowserEvents.getEventOptions( passiveEvents, true ) );
  }

  /**
   * Returns the value to provide as the 3rd parameter to addEventListener/removeEventListener.
   *
   * @param isMain - If false, it is used on the "document" for workarounds.
   */
  private static getEventOptions( passiveEvents: boolean | null, isMain: boolean ): { passive: boolean; capture?: boolean } | boolean {
    const passDirectPassiveFlag = Features.passive && passiveEvents !== null;
    if ( !passDirectPassiveFlag ) {
      return false;
    }
    else {
      const eventOptions: { passive: boolean; capture?: boolean } = { passive: passiveEvents };
      if ( isMain ) {
        eventOptions.capture = false;
      }

      assert && assert( !eventOptions.capture, 'Do not use capture without consulting globalKeyStateTracker, ' +
                                               'which expects have listeners called FIRST in keyboard-related cases.' );
      return eventOptions;
    }
  }

  /**
   * Will be checked/mutated when listeners are added/removed.
   */
  private static listenersAttachedToWindow = 0;

  /**
   * Will be checked/mutated when listeners are added/removed.
   */
  private static listenersAttachedToElement = 0;

  /**
   * All Displays that should have input events forwarded.
   */
  private static attachedDisplays: Display[] = [];

  /**
   * Whether pointer events in the format specified by the MS specification are allowed.
   */
  private static canUseMSPointerEvents =
    window.navigator &&
    // @ts-expect-error msPointerEnabled should exist
    window.navigator.msPointerEnabled;

  /**
   * All W3C pointer event types that we care about.
   */
  private static pointerListenerTypes = [
    'pointerdown',
    'pointerup',
    'pointermove',
    'pointerover',
    'pointerout',
    'pointercancel',
    'gotpointercapture',
    'lostpointercapture'
  ];

  /**
   * All MS pointer event types that we care about.
   */
  private static msPointerListenerTypes = [
    'MSPointerDown',
    'MSPointerUp',
    'MSPointerMove',
    'MSPointerOver',
    'MSPointerOut',
    'MSPointerCancel'
  ];

  /**
   * All touch event types that we care about
   */
  private static touchListenerTypes = [
    'touchstart',
    'touchend',
    'touchmove',
    'touchcancel'
  ];

  /**
   * All mouse event types that we care about
   */
  private static mouseListenerTypes = [
    'mousedown',
    'mouseup',
    'mousemove',
    'mouseover',
    'mouseout'
  ];

  /**
   * All wheel event types that we care about
   */
  private static wheelListenerTypes = [
    'wheel'
  ];

  /**
   * Alternative input types
   */
  private static altListenerTypes = PDOMUtils.DOM_EVENTS;

  /**
   * Returns all event types that will be listened to on this specific platform.
   */
  private static getNonWheelUsedTypes( listeningToWindow: boolean ): string[] {
    let eventTypes;

    // Whether pointer events in the format specified by the W3C specification are allowed.
    // NOTE: We used to disable this for Firefox, but with skipping preventDefault()
    // on pointer events, it seems to work fine.
    // See https://github.com/phetsims/scenery/issues/837 and
    // https://github.com/scenerystack/scenerystack/issues/42 for reference.
    // @ts-expect-error pointerEnabled should exist
    const canUsePointerEvents = !!( ( window.navigator && window.navigator.pointerEnabled ) || window.PointerEvent );

    if ( canUsePointerEvents ) {
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
  }

  /**
   * Connects event listeners directly to the window.
   *
   * @param passiveEvents - The value of the `passive` option for adding/removing DOM event listeners
   */
  private static connectWindowListeners( passiveEvents: boolean | null ): void {
    this.addOrRemoveListeners( window, true, passiveEvents );
  }

  /**
   * Disconnects event listeners from the window.
   *
   * @param passiveEvents - The value of the `passive` option for adding/removing DOM event listeners
   */
  private static disconnectWindowListeners( passiveEvents: boolean | null ): void {
    this.addOrRemoveListeners( window, false, passiveEvents );
  }

  /**
   * Either adds or removes event listeners to an object, depending on the flag.
   *
   * @param element - The element (window or DOM element) to add listeners to.
   * @param addOrRemove - If true, listeners will be added. If false, listeners will be removed.
   * @param passiveEvents - The value of the `passive` option for adding/removing DOM event listeners
   *                                       NOTE: if it is passed in as null, the default value for the browser will be
   *                                       used.
   */
  private static addOrRemoveListeners( element: Element | typeof window, addOrRemove: boolean, passiveEvents: boolean | null ): void {
    const forWindow = element === window;
    assert && assert( !forWindow || ( BrowserEvents.listenersAttachedToWindow > 0 ) === !addOrRemove,
      'Do not add listeners to the window when already attached, or remove listeners when none are attached' );

    const delta = addOrRemove ? 1 : -1;
    if ( forWindow ) {
      BrowserEvents.listenersAttachedToWindow += delta;
    }
    else {
      BrowserEvents.listenersAttachedToElement += delta;
    }
    assert && assert( BrowserEvents.listenersAttachedToWindow === 0 || BrowserEvents.listenersAttachedToElement === 0,
      'Listeners should not be added both with addDisplayToWindow and addDisplayToElement. Use only one.' );

    const method = addOrRemove ? 'addEventListener' : 'removeEventListener';

    // {Array.<string>}
    const eventTypes = this.getNonWheelUsedTypes( element === window );

    for ( let i = 0; i < eventTypes.length; i++ ) {
      const type = eventTypes[ i ];

      // If we add input listeners to the window itself, iOS Safari 7 won't send touch events to displays in an
      // iframe unless we also add dummy listeners to the document.
      if ( forWindow ) {
        // Workaround for older browsers needed,
        // see https://developer.mozilla.org/en-US/docs/Web/API/EventTarget/addEventListener#Improving_scrolling_performance_with_passive_listeners
        document[ method ]( type, noop, BrowserEvents.getEventOptions( passiveEvents, false ) );
      }

      // @ts-expect-error Trust us on this
      const callback = BrowserEvents[ `on${type}` ];
      assert && assert( !!callback );

      // Workaround for older browsers needed,
      // see https://developer.mozilla.org/en-US/docs/Web/API/EventTarget/addEventListener#Improving_scrolling_performance_with_passive_listeners
      element[ method ]( type, callback, BrowserEvents.getEventOptions( passiveEvents, true ) );
    }
  }

  /**
   * Sets an event from the window to be batched on all of the displays.
   *
   * @param eventContext
   * @param batchType - TODO: turn to full enumeration? https://github.com/phetsims/scenery/issues/1581
   * @param inputCallbackName - e.g. 'mouseDown', will trigger Input.mouseDown
   * @param triggerImmediate - Whether this will be force-executed now, causing all batched events to fire.
   *                           Useful for events (like mouseup) that responding synchronously is
   *                           necessary for certain security-sensitive actions (like triggering
   *                           full-screen).
   */
  private static batchWindowEvent(
    eventContext: EventContext,
    batchType: BatchedDOMEventType,
    inputCallbackName: string,
    triggerImmediate: boolean
  ): void {
    // NOTE: For now, we don't check whether the event is actually within the display's boundingClientRect. Most
    // displays will want to receive events outside of their bounds (especially for checking drags and mouse-ups
    // outside of their bounds).
    for ( let i = 0; i < BrowserEvents.attachedDisplays.length; i++ ) {
      const display = BrowserEvents.attachedDisplays[ i ];
      const input = display._input!;

      // Filter out events that are specific to another Display's DOM element
      // See https://github.com/scenerystack/scenerystack/issues/42
      if (
        eventContext.domEvent.currentTarget &&
        eventContext.domEvent.currentTarget !== window &&
        eventContext.domEvent.currentTarget !== display.domElement
      ) {
        continue;
      }

      if ( !BrowserEvents.blockFocusCallbacks || ( inputCallbackName !== 'focusIn' && inputCallbackName !== 'focusOut' ) ) {
        // @ts-expect-error Method should exist
        input.batchEvent( eventContext, batchType, input[ inputCallbackName ], triggerImmediate );
      }
    }
  }

  /**
   * Listener for window's pointerdown event.
   */
  private static onpointerdown( domEvent: Event ): void {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'pointerdown' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // Get the active element BEFORE any actions are taken
    const eventContext = new EventContext( domEvent );

    if ( ( domEvent as PointerEvent ).pointerType === 'mouse' ) {
      DisplayGlobals.userGestureEmitter.emit();
    }

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( eventContext, BatchedDOMEventType.POINTER_TYPE, 'pointerDown', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  }

  /**
   * Listener for window's pointerup event.
   */
  private static onpointerup( domEvent: Event ): void {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'pointerup' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // Get the active element BEFORE any actions are taken
    const eventContext = new EventContext( domEvent );

    DisplayGlobals.userGestureEmitter.emit();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( eventContext, BatchedDOMEventType.POINTER_TYPE, 'pointerUp', true );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  }

  /**
   * Listener for window's pointermove event.
   */
  private static onpointermove( domEvent: Event ): void {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'pointermove' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.POINTER_TYPE, 'pointerMove', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  }

  /**
   * Listener for window's pointerover event.
   */
  private static onpointerover( domEvent: Event ): void {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'pointerover' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.POINTER_TYPE, 'pointerOver', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  }

  /**
   * Listener for window's pointerout event.
   */
  private static onpointerout( domEvent: Event ): void {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'pointerout' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.POINTER_TYPE, 'pointerOut', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  }

  /**
   * Listener for window's pointercancel event.
   */
  private static onpointercancel( domEvent: Event ): void {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'pointercancel' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.POINTER_TYPE, 'pointerCancel', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  }

  /**
   * Listener for window's gotpointercapture event.
   */
  private static ongotpointercapture( domEvent: Event ): void {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'gotpointercapture' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.POINTER_TYPE, 'gotPointerCapture', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  }

  /**
   * Listener for window's lostpointercapture event.
   */
  private static onlostpointercapture( domEvent: Event ): void {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'lostpointercapture' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.POINTER_TYPE, 'lostPointerCapture', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  }

  /**
   * Listener for window's MSPointerDown event.
   */
  private static onMSPointerDown( domEvent: Event ): void {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'MSPointerDown' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.MS_POINTER_TYPE, 'pointerDown', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  }

  /**
   * Listener for window's MSPointerUp event.
   */
  private static onMSPointerUp( domEvent: Event ): void {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'MSPointerUp' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.MS_POINTER_TYPE, 'pointerUp', true );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  }

  /**
   * Listener for window's MSPointerMove event.
   */
  private static onMSPointerMove( domEvent: Event ): void {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'MSPointerMove' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.MS_POINTER_TYPE, 'pointerMove', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  }

  /**
   * Listener for window's MSPointerOver event.
   */
  private static onMSPointerOver( domEvent: Event ): void {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'MSPointerOver' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.MS_POINTER_TYPE, 'pointerOver', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  }

  /**
   * Listener for window's MSPointerOut event.
   */
  private static onMSPointerOut( domEvent: Event ): void {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'MSPointerOut' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.MS_POINTER_TYPE, 'pointerOut', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  }

  /**
   * Listener for window's MSPointerCancel event.
   */
  private static onMSPointerCancel( domEvent: Event ): void {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'MSPointerCancel' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.MS_POINTER_TYPE, 'pointerCancel', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  }

  /**
   * Listener for window's touchstart event.
   */
  private static ontouchstart( domEvent: Event ): void {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'touchstart' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.TOUCH_TYPE, 'touchStart', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  }

  /**
   * Listener for window's touchend event.
   */
  private static ontouchend( domEvent: Event ): void {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'touchend' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // Get the active element BEFORE any actions are taken
    const eventContext = new EventContext( domEvent );

    DisplayGlobals.userGestureEmitter.emit();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( eventContext, BatchedDOMEventType.TOUCH_TYPE, 'touchEnd', true );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  }

  /**
   * Listener for window's touchmove event.
   */
  private static ontouchmove( domEvent: Event ): void {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'touchmove' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.TOUCH_TYPE, 'touchMove', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  }

  /**
   * Listener for window's touchcancel event.
   */
  private static ontouchcancel( domEvent: Event ): void {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'touchcancel' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.TOUCH_TYPE, 'touchCancel', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  }

  /**
   * Listener for window's mousedown event.
   */
  private static onmousedown( domEvent: Event ): void {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'mousedown' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // Get the active element BEFORE any actions are taken
    const eventContext = new EventContext( domEvent );

    DisplayGlobals.userGestureEmitter.emit();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( eventContext, BatchedDOMEventType.MOUSE_TYPE, 'mouseDown', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  }

  /**
   * Listener for window's mouseup event.
   */
  private static onmouseup( domEvent: Event ): void {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'mouseup' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // Get the active element BEFORE any actions are taken
    const eventContext = new EventContext( domEvent );

    DisplayGlobals.userGestureEmitter.emit();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( eventContext, BatchedDOMEventType.MOUSE_TYPE, 'mouseUp', true );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  }

  /**
   * Listener for window's mousemove event.
   */
  private static onmousemove( domEvent: Event ): void {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'mousemove' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.MOUSE_TYPE, 'mouseMove', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  }

  /**
   * Listener for window's mouseover event.
   */
  private static onmouseover( domEvent: Event ): void {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'mouseover' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.MOUSE_TYPE, 'mouseOver', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  }

  /**
   * Listener for window's mouseout event.
   */
  private static onmouseout( domEvent: Event ): void {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'mouseout' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.MOUSE_TYPE, 'mouseOut', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  }

  /**
   * Listener for window's wheel event.
   */
  private static onwheel( domEvent: Event ): void {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'wheel' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.WHEEL_TYPE, 'wheel', false );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  }

  public static onfocusin( domEvent: Event ): void {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'focusin' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.

    BrowserEvents.dispatchingFocusEvents = true;

    // Update state related to focus immediately and allowing for reentrancy for focus state
    // that must match the browser's focus state.
    FocusManager.updatePDOMFocusFromEvent( BrowserEvents.attachedDisplays, domEvent as FocusEvent, true );

    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.ALT_TYPE, 'focusIn', true );

    BrowserEvents.dispatchingFocusEvents = false;

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  }

  public static onfocusout( domEvent: Event ): void {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'focusout' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.

    BrowserEvents.dispatchingFocusEvents = true;

    // Update state related to focus immediately and allowing for reentrancy for focus state
    FocusManager.updatePDOMFocusFromEvent( BrowserEvents.attachedDisplays, domEvent as FocusEvent, false );

    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.ALT_TYPE, 'focusOut', true );

    BrowserEvents.dispatchingFocusEvents = false;

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  }

  public static oninput( domEvent: Event ): void {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'input' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.

    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.ALT_TYPE, 'input', true );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  }

  public static onchange( domEvent: Event ): void {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'change' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.ALT_TYPE, 'change', true );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  }

  public static onclick( domEvent: Event ): void {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'click' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.ALT_TYPE, 'click', true );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  }

  public static onkeydown( domEvent: Event ): void {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'keydown' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.ALT_TYPE, 'keyDown', true );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  }

  public static onkeyup( domEvent: Event ): void {
    sceneryLog && sceneryLog.OnInput && sceneryLog.OnInput( 'keyup' );
    sceneryLog && sceneryLog.OnInput && sceneryLog.push();

    // NOTE: Will be called without a proper 'this' reference. Do NOT rely on it here.
    BrowserEvents.batchWindowEvent( new EventContext( domEvent ), BatchedDOMEventType.ALT_TYPE, 'keyUp', true );

    sceneryLog && sceneryLog.OnInput && sceneryLog.pop();
  }
}

scenery.register( 'BrowserEvents', BrowserEvents );