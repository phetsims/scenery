// Copyright 2019-2022, University of Colorado Boulder

/**
 * A general type for keyboard dragging. Objects can be dragged in two dimensions with the arrow keys and with the WASD
 * keys. This can be added to a node through addInputListener for accessibility, which is mixed into Nodes with
 * the ParallelDOM trait.
 *
 * JavaScript does not natively handle multiple 'keydown' events at once, so we have a custom implementation that
 * tracks which keys are down and for how long in a step() function. To support keydown timing, AXON/timer is used. In
 * scenery this is supported via Display.updateOnRequestAnimationFrame(), which will step the time on each frame.
 * If using KeyboardDragListener in a more customized Display, like done in phetsims (see JOIST/Sim), the time must be
 * manually stepped (by emitting the timer).
 *
 * For the purposes of this file, a "hotkey" is a collection of keys that, when pressed together in the right
 * order, fire a callback.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 * @author Michael Barlow
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */

import PhetioAction from '../../../tandem/js/PhetioAction.js';
import EnabledComponent, { EnabledComponentOptions } from '../../../axon/js/EnabledComponent.js';
import Emitter from '../../../axon/js/Emitter.js';
import Property from '../../../axon/js/Property.js';
import stepTimer from '../../../axon/js/stepTimer.js';
import Bounds2 from '../../../dot/js/Bounds2.js';
import Transform3 from '../../../dot/js/Transform3.js';
import Vector2 from '../../../dot/js/Vector2.js';
import platform from '../../../phet-core/js/platform.js';
import EventType from '../../../tandem/js/EventType.js';
import Tandem from '../../../tandem/js/Tandem.js';
import { IInputListener, KeyboardUtils, Node, PDOMPointer, scenery, SceneryEvent } from '../imports.js';
import IProperty from '../../../axon/js/IProperty.js';
import optionize from '../../../phet-core/js/optionize.js';
import IReadOnlyProperty from '../../../axon/js/IReadOnlyProperty.js';

type PressedKeyTiming = {

  // Is the key currently down?
  keyDown: boolean;

  // How long has the key been pressed in milliseconds
  timeDown: number;

  // KeyboardEvent.key string
  key: string;
};

type Hotkey = {

  // Keys to be pressed in order to trigger the callback of the Hotkey
  keys: string[];

  // Called when keys are pressed in order
  callback: () => void;
};

type SelfOptions = {

  // While direction key is down, this will be the 1D velocity for movement. The position will
  // change this much in view coordinates every second.
  dragVelocity?: number;

  // If shift key down while pressing direction key, this will be the 1D delta for movement in view
  // coordinates every second. Must be less than or equal to dragVelocity, and it is intended to provide user with
  // more fine-grained control of motion.
  shiftDragVelocity?: number;

  // If provided, it will be synchronized with the drag position in the model frame, applying provided transforms as
  // needed. Most useful when used with transform option
  positionProperty?: IProperty<Vector2> | null;

  // If provided, this will be the conversion between the view and model coordinate frames. Usually most useful when
  // paired with the positionProperty.
  transform?: Transform3 | null;

  // If provided, the model position will be constrained to be inside these bounds, in model coordinates
  dragBoundsProperty?: IReadOnlyProperty<Bounds2 | null> | null;

  // Called when keyboard drag is started (on initial press).
  start?: ( ( event: SceneryEvent ) => void ) | null;

  // Called during drag. Note that this does not provide the SceneryEvent. Dragging happens during animation
  // (as long as keys are down), so there is no event associated with the drag.
  drag?: ( ( viewDelta: Vector2 ) => void ) | null;

  // Called when keyboard dragging ends.
  end?: ( ( event: SceneryEvent ) => void ) | null;

  // Arrow keys must be pressed this long to begin movement set on moveOnHoldInterval, in ms
  moveOnHoldDelay?: number;

  // Time interval at which the object will change position while the arrow key is held down, in ms
  moveOnHoldInterval?: number;

  // On initial key press, how much the position Property will change in view coordinates, generally only needed when
  // there is a moveOnHoldDelay or moveOnHoldInterval. In ms.
  downDelta?: number;

  // The amount PositionProperty changes in view coordinates, generally only needed when there is a moveOnHoldDelay or
  // moveOnHoldInterval. In ms.
  shiftDownDelta?: number;

  // Time interval at which holding down a hotkey group will trigger an associated listener, in ms
  hotkeyHoldInterval?: number;

  // EnabledComponent
  // By default, do not instrument the enabledProperty; opt in with this option. See EnabledComponent
  phetioEnabledPropertyInstrumented?: boolean;

  // phet-io
  tandem?: Tandem;

  // Though DragListener is not instrumented, declare these here to support properly passing this to children, see
  // https://github.com/phetsims/tandem/issues/60.
  phetioReadOnly?: boolean;
};

export type KeyboardDragListenerOptions = SelfOptions & EnabledComponentOptions;

class KeyboardDragListener extends EnabledComponent implements IInputListener {

  // See options for documentation
  private _start: ( ( event: SceneryEvent ) => void ) | null;
  private _drag: ( ( viewDelta: Vector2 ) => void ) | null;
  private _end: ( ( event: SceneryEvent ) => void ) | null;
  private _dragBoundsProperty: IReadOnlyProperty<Bounds2 | null>;
  private _transform: Transform3 | null;
  private _positionProperty: IProperty<Vector2> | null;
  private _dragVelocity: number;
  private _shiftDragVelocity: number;
  private _downDelta: number;
  private _shiftDownDelta: number;
  private _moveOnHoldDelay: number;
  private _moveOnHoldInterval: number;
  private _hotkeyHoldInterval: number;

  // Tracks the state of the keyboard. JavaScript doesn't handle multiple key presses, so we track which keys are
  // currently down and update based on state of this collection of objects.
  // TODO: Consider a global state object for this, see https://github.com/phetsims/scenery/issues/1054
  private keyState: PressedKeyTiming[];

  // A list of hotkeys, each of which has some behavior when each individual key of the hotkey is pressed in order.
  // See this.addHotkey() for more information.
  private _hotkeys: Hotkey[];

  // The Hotkey that is currently down
  private currentHotkey: Hotkey | null;

  // When a hotkey group is pressed down, dragging will be disabled until any keys are up again
  private hotkeyDisablingDragging: boolean;

  // Delay before calling a Hotkey listener (if all Hotkeys are being held down), incremented in step. In milliseconds.
  private hotkeyHoldIntervalCounter: number;

  // Counters to allow for press-and-hold functionality that enables user to incrementally move the draggable object or
  // hold the movement key for continuous or stepped movement - values in ms
  private moveOnHoldDelayCounter: number;
  private moveOnHoldIntervalCounter: number;

  // Variable to determine when the initial delay is complete
  private delayComplete: boolean;

  // Fires to conduct the start and end of a drag, added for PhET-iO interoperability
  private dragStartAction: PhetioAction<[ SceneryEvent ]>;
  private dragEndAction: PhetioAction<[ SceneryEvent ]>;

  // @deprecated - Use the drag option instead.
  public dragEmitter: Emitter;

  // Implements disposal
  private readonly _disposeKeyboardDragListener: () => void;

  // A listener added to the pointer when dragging starts so that we can attach a listener and provide a channel of
  // communication to the AnimatedPanZoomListener to define custom behavior for screen panning during a drag operation.
  private readonly _pointerListener: IInputListener;

  // A reference to the Pointer during a drag operation so that we can add/remove the _pointerListener.
  private _pointer: PDOMPointer | null;

  constructor( providedOptions?: KeyboardDragListenerOptions ) {

    const options = optionize<KeyboardDragListenerOptions, SelfOptions, EnabledComponentOptions>()( {
      dragVelocity: 600,
      shiftDragVelocity: 300,
      positionProperty: null,
      transform: null,
      dragBoundsProperty: null,
      start: null,
      drag: null,
      end: null,
      moveOnHoldDelay: 0,
      moveOnHoldInterval: 0,
      downDelta: 0,
      shiftDownDelta: 0,
      hotkeyHoldInterval: 800,
      phetioEnabledPropertyInstrumented: false,
      tandem: Tandem.REQUIRED,

      // DragListener by default doesn't allow PhET-iO to trigger drag Action events
      phetioReadOnly: true
    }, providedOptions );

    assert && assert( options.shiftDragVelocity <= options.dragVelocity, 'shiftDragVelocity should be less than or equal to shiftDragVelocity, it is intended to provide more fine-grained control' );

    super( options );

    // mutable attributes declared from options, see options for info, as well as getters and setters
    this._start = options.start;
    this._drag = options.drag;
    this._end = options.end;
    this._dragBoundsProperty = ( options.dragBoundsProperty || new Property( null ) );
    this._transform = options.transform;
    this._positionProperty = options.positionProperty;
    this._dragVelocity = options.dragVelocity;
    this._shiftDragVelocity = options.shiftDragVelocity;
    this._downDelta = options.downDelta;
    this._shiftDownDelta = options.shiftDownDelta;
    this._moveOnHoldDelay = options.moveOnHoldDelay;
    this._moveOnHoldInterval = options.moveOnHoldInterval;
    this._hotkeyHoldInterval = options.hotkeyHoldInterval;

    this.keyState = [];
    this._hotkeys = [];
    this.currentHotkey = null;
    this.hotkeyDisablingDragging = false;

    // This is initialized to the "threshold" so that the first hotkey will fire immediately. Only subsequent actions
    // while holding the hotkey should result in a delay of this much. in ms
    this.hotkeyHoldIntervalCounter = this._hotkeyHoldInterval;

    this.moveOnHoldDelayCounter = 0;
    this.moveOnHoldIntervalCounter = 0;

    this.delayComplete = false;

    this.dragStartAction = new PhetioAction( event => {
      const key = KeyboardUtils.getEventCode( event.domEvent );
      assert && assert( key, 'How can we have a null key for KeyboardDragListener?' );

      // If there are no movement keys down, attach a listener to the Pointer that will tell the AnimatedPanZoomListener
      // to keep this Node in view
      if ( !this.movementKeysDown && KeyboardUtils.isMovementKey( event.domEvent ) ) {
        assert && assert( this._pointer === null, 'We should have cleared the Pointer reference by now.' );
        this._pointer = event.pointer as PDOMPointer;
        event.pointer.addInputListener( this._pointerListener, true );
      }

      // update the key state
      this.keyState.push( {
        keyDown: true,
        key: key!,
        timeDown: 0 // in ms
      } );

      if ( this._start ) {
        if ( this.movementKeysDown ) {
          this._start( event );
        }
      }

      // move object on first down before a delay
      const positionDelta = this.shiftKeyDown() ? this._shiftDownDelta : this._downDelta;
      this.updatePosition( positionDelta );
    }, {
      parameters: [ { name: 'event', phetioType: SceneryEvent.SceneryEventIO } ],
      tandem: options.tandem.createTandem( 'dragStartAction' ),
      phetioDocumentation: 'Emits whenever a keyboard drag starts.',
      phetioReadOnly: options.phetioReadOnly,
      phetioEventType: EventType.USER
    } );

    // Emits an event every drag
    // @deprecated - Use the drag option instead
    this.dragEmitter = new Emitter( {
      tandem: options.tandem.createTandem( 'dragEmitter' ),
      phetioHighFrequency: true,
      phetioDocumentation: 'Emits whenever a keyboard drag occurs.',
      phetioReadOnly: options.phetioReadOnly,
      phetioEventType: EventType.USER
    } );

    this.dragEndAction = new PhetioAction( event => {

      // If there are no movement keys down, attach a listener to the Pointer that will tell the AnimatedPanZoomListener
      // to keep this Node in view
      if ( !this.movementKeysDown ) {
        assert && assert( event.pointer === this._pointer, 'How could the event Pointer be anything other than this PDOMPointer?' );
        this._pointer!.removeInputListener( this._pointerListener );
        this._pointer = null;
      }

      this._end && this._end( event );
    }, {
      parameters: [ { name: 'event', phetioType: SceneryEvent.SceneryEventIO } ],
      tandem: options.tandem.createTandem( 'dragEndAction' ),
      phetioDocumentation: 'Emits whenever a keyboard drag ends.',
      phetioReadOnly: options.phetioReadOnly,
      phetioEventType: EventType.USER
    } );

    // step the drag listener, must be removed in dispose
    const stepListener = this.step.bind( this );
    stepTimer.addListener( stepListener );

    this.enabledProperty.lazyLink( this.onEnabledPropertyChange.bind( this ) );

    this._pointerListener = {
      listener: this,
      interrupt: this.interrupt.bind( this )
    };

    this._pointer = null;

    // called in dispose
    this._disposeKeyboardDragListener = () => {
      stepTimer.removeListener( stepListener );
    };
  }

  /**
   * Returns the drag bounds in model coordinates.
   */
  public getDragBounds(): Bounds2 | null {
    return this._dragBoundsProperty.value;
  }

  get dragBounds() { return this.getDragBounds(); }

  /**
   * Sets the drag transform of the listener.
   */
  public setTransform( transform: Transform3 | null ): void {
    this._transform = transform;
  }

  set transform( transform ) { this.setTransform( transform ); }

  get transform() { return this.getTransform(); }

  /**
   * Returns the transform of the listener.
   */
  public getTransform(): Transform3 | null {
    return this._transform;
  }

  /**
   * Getter for the dragVelocity property, see options.dragVelocity for more info.
   */
  get dragVelocity(): number { return this._dragVelocity; }

  /**
   * Setter for the dragVelocity property, see options.dragVelocity for more info.
   */
  set dragVelocity( dragVelocity: number ) { this._dragVelocity = dragVelocity; }

  /**
   * Getter for the shiftDragVelocity property, see options.shiftDragVelocity for more info.
   */
  get shiftDragVelocity(): number { return this._shiftDragVelocity; }

  /**
   * Setter for the shiftDragVelocity property, see options.shiftDragVelocity for more info.
   */
  set shiftDragVelocity( shiftDragVelocity: number ) { this._shiftDragVelocity = shiftDragVelocity; }

  /**
   * Getter for the downDelta property, see options.downDelta for more info.
   */
  get downDelta(): number { return this._downDelta; }

  /**
   * Setter for the downDelta property, see options.downDelta for more info.
   */
  set downDelta( downDelta: number ) { this._downDelta = downDelta; }

  /**
   * Getter for the shiftDownDelta property, see options.shiftDownDelta for more info.
   */
  get shiftDownDelta(): number { return this._shiftDownDelta; }

  /**
   * Setter for the shiftDownDelta property, see options.shiftDownDelta for more info.
   * @param {number|null} shiftDownDelta
   */
  set shiftDownDelta( shiftDownDelta: number ) { this._shiftDownDelta = shiftDownDelta; }

  /**
   * Getter for the moveOnHoldDelay property, see options.moveOnHoldDelay for more info.
   */
  get moveOnHoldDelay(): number { return this._moveOnHoldDelay; }

  /**
   * Setter for the moveOnHoldDelay property, see options.moveOnHoldDelay for more info.
   */
  set moveOnHoldDelay( moveOnHoldDelay: number ) { this._moveOnHoldDelay = moveOnHoldDelay; }

  /**
   * Getter for the moveOnHoldInterval property, see options.moveOnHoldInterval for more info.
   */
  get moveOnHoldInterval(): number { return this._moveOnHoldInterval; }

  /**
   * Setter for the moveOnHoldInterval property, see options.moveOnHoldInterval for more info.
   */
  set moveOnHoldInterval( moveOnHoldInterval: number ) { this._moveOnHoldInterval = moveOnHoldInterval; }

  /**
   * Getter for the hotkeyHoldInterval property, see options.hotkeyHoldInterval for more info.
   */
  get hotkeyHoldInterval(): number { return this._hotkeyHoldInterval; }

  /**
   * Setter for the hotkeyHoldInterval property, see options.hotkeyHoldInterval for more info.
   */
  set hotkeyHoldInterval( hotkeyHoldInterval: number ) { this._hotkeyHoldInterval = hotkeyHoldInterval; }

  get isPressed(): boolean {
    return !!this._pointer;
  }

  /**
   * Get the current target Node of the drag.
   */
  public getCurrentTarget(): Node {
    assert && assert( this.isPressed, 'We have no currentTarget if we are not pressed' );
    assert && assert( this._pointer && this._pointer.trail, 'Must have a Pointer with an active trail if we are pressed' );
    return this._pointer!.trail!.lastNode();
  }

  /**
   * Fired when the enabledProperty changes
   */
  private onEnabledPropertyChange( enabled: boolean ): void {
    !enabled && this.interrupt();
  }

  /**
   * Implements keyboard dragging when listener is attached to the Node, public because this is called as part of
   * the Scenery Input API, but clients should not call this directly.
   */
  public keydown( event: SceneryEvent ): void {
    const domEvent = event.domEvent as KeyboardEvent;
    const key = KeyboardUtils.getEventCode( domEvent );
    assert && assert( key, 'How can we have a null key from a keydown in KeyboardDragListener?' );

    // If the meta key is down (command key/windows key) prevent movement and do not preventDefault.
    // Meta key + arrow key is a command to go back a page, and we need to allow that. But also, macOS
    // fails to provide keyup events once the meta key is pressed, see
    // http://web.archive.org/web/20160304022453/http://bitspushedaround.com/on-a-few-things-you-may-not-know-about-the-hellish-command-key-and-javascript-events/
    if ( domEvent.metaKey ) {
      return;
    }

    // required to work with Safari and VoiceOver, otherwise arrow keys will move virtual cursor, see https://github.com/phetsims/balloons-and-static-electricity/issues/205#issuecomment-263428003
    // prevent default for WASD too, see https://github.com/phetsims/friction/issues/167
    if ( KeyboardUtils.isMovementKey( domEvent ) ) {
      domEvent.preventDefault();
    }

    // reserve keyboard events for dragging to prevent default panning behavior with zoom features
    event.pointer.reserveForKeyboardDrag();

    // if the key is already down, don't do anything else (we don't want to create a new keystate object
    // for a key that is already being tracked and down, nor call startDrag every keydown event)
    if ( this.keyInListDown( [ key! ] ) ) { return; }

    // Prevent a VoiceOver bug where pressing multiple arrow keys at once causes the AT to send the wrong keys
    // through the keyup event - as a workaround, we only allow one arrow key to be down at a time. If two are pressed
    // down, we immediately clear the keystate and return
    // see https://github.com/phetsims/balloons-and-static-electricity/issues/384
    if ( platform.safari ) {
      if ( KeyboardUtils.isArrowKey( domEvent ) ) {
        if ( this.keyInListDown( [
          KeyboardUtils.KEY_RIGHT_ARROW, KeyboardUtils.KEY_LEFT_ARROW,
          KeyboardUtils.KEY_UP_ARROW, KeyboardUtils.KEY_DOWN_ARROW ] ) ) {
          this.interrupt();
          return;
        }
      }
    }

    this.canDrag() && this.dragStartAction.execute( event );
  }

  /**
   * Behavior for keyboard 'up' DOM event. Public so it can be attached with addInputListener()
   *
   * Note that this event is assigned in the constructor, and not to the prototype. As of writing this,
   * `Node.addInputListener` only supports type properties as event listeners, and not the event keys as
   * prototype methods. Please see https://github.com/phetsims/scenery/issues/851 for more information.
   *
   * @param {SceneryEvent} event
   */
  public keyup( event: SceneryEvent ): void {
    const domEvent = event.domEvent as KeyboardEvent;
    const key = KeyboardUtils.getEventCode( domEvent );

    const moveKeysDown = this.movementKeysDown;

    // if the shift key is down when we navigate to the object, add it to the keystate because it won't be added until
    // the next keydown event
    if ( key === KeyboardUtils.KEY_TAB ) {
      if ( domEvent.shiftKey ) {

        // add 'shift' to the keystate until it is released again
        this.keyState.push( {
          keyDown: true,
          key: KeyboardUtils.KEY_SHIFT_LEFT,
          timeDown: 0 // in ms
        } );
      }
    }

    for ( let i = 0; i < this.keyState.length; i++ ) {
      if ( key === this.keyState[ i ].key ) {
        this.keyState.splice( i, 1 );
      }
    }

    const moveKeysStillDown = this.movementKeysDown;

    // if movement keys are no longer down after keyup, call the optional end drag function
    if ( !moveKeysStillDown && moveKeysDown !== moveKeysStillDown ) {
      this.dragEndAction.execute( event );
    }

    // if any current hotkey keys are no longer down, clear out the current hotkey and reset.
    if ( this.currentHotkey && !this.allKeysInListDown( this.currentHotkey.keys ) ) {
      this.resetHotkeyState();
    }

    this.resetPressAndHold();
  }

  /**
   * Interrupts and resets the listener on blur so that listener state is reset and keys are removed from the keyState
   * array. Public because this is called with the scenery listener API. Clients should not call this directly.
   */
  public blur( event: SceneryEvent ): void {
    this.interrupt();
  }

  /**
   * Step function for the drag handler. JavaScript does not natively handle multiple keydown events at once,
   * so we need to track the state of the keyboard in an Object and manage dragging in this function.
   * In order for the drag handler to work.
   *
   * @param dt - in seconds
   */
  private step( dt: number ): void {
    const ms = dt * 1000;

    // no-op unless a key is down
    if ( this.keyState.length > 0 ) {
      // for each key that is still down, increment the tracked time that has been down
      for ( let i = 0; i < this.keyState.length; i++ ) {
        if ( this.keyState[ i ].keyDown ) {
          this.keyState[ i ].timeDown += ms;
        }
      }

      // dt is in seconds and we convert to ms
      this.moveOnHoldDelayCounter += ms;
      this.moveOnHoldIntervalCounter += ms;

      // update timer for keygroup if one is being held down
      if ( this.currentHotkey ) {
        this.hotkeyHoldIntervalCounter += ms;
      }

      // calculate change in position from time step
      const positionVelocitySeconds = this.shiftKeyDown() ? this._shiftDragVelocity : this._dragVelocity;
      const positionVelocityMilliseconds = positionVelocitySeconds / 1000;
      const positionDelta = ms * positionVelocityMilliseconds;

      if ( this.moveOnHoldDelayCounter >= this._moveOnHoldDelay && !this.delayComplete ) {
        this.updatePosition( positionDelta );
        this.delayComplete = true;
      }

      if ( this.delayComplete && this.moveOnHoldIntervalCounter >= this._moveOnHoldInterval ) {
        this.updatePosition( positionDelta );
      }
    }
  }

  /**
   * Returns true if a drag can begin from input with this listener.
   */
  private canDrag(): boolean {
    return this.enabledProperty.value;
  }

  /**
   * Update the state of hotkeys, and fire hotkey callbacks if one is active.
   */
  private updateHotkeys(): void {

    // check to see if any hotkey combinations are down
    for ( let j = 0; j < this._hotkeys.length; j++ ) {
      const hotkeysDownList = [];
      const keys = this._hotkeys[ j ].keys;

      for ( let k = 0; k < keys.length; k++ ) {
        for ( let l = 0; l < this.keyState.length; l++ ) {
          if ( this.keyState[ l ].key === keys[ k ] ) {
            hotkeysDownList.push( this.keyState[ l ] );
          }
        }
      }

      // There is only a single hotkey and it is down, the hotkeys must be in order
      let keysInOrder = hotkeysDownList.length === 1 && keys.length === 1;

      // the hotkeysDownList array order should match the order of the key group, so now we just need to make
      // sure that the key down times are in the right order
      for ( let m = 0; m < hotkeysDownList.length - 1; m++ ) {
        if ( hotkeysDownList[ m + 1 ] && hotkeysDownList[ m ].timeDown > hotkeysDownList[ m + 1 ].timeDown ) {
          keysInOrder = true;
        }
      }

      // if keys are in order, call the callback associated with the group, and disable dragging until
      // all hotkeys associated with that group are up again
      if ( keysInOrder ) {
        this.currentHotkey = this._hotkeys[ j ];
        if ( this.hotkeyHoldIntervalCounter >= this._hotkeyHoldInterval ) {

          // Set the counter to begin counting the next interval between hotkey activations.
          this.hotkeyHoldIntervalCounter = 0;

          // call the callback last, after internal state has been updated. This solves a bug caused if this callback
          // then makes this listener interrupt.
          this._hotkeys[ j ].callback();
        }
      }
    }

    // if a key group is down, check to see if any of those keys are still down - if so, we will disable dragging
    // until all of them are up
    if ( this.currentHotkey ) {
      if ( this.keyInListDown( this.currentHotkey.keys ) ) {
        this.hotkeyDisablingDragging = true;
      }
      else {
        this.hotkeyDisablingDragging = false;

        // keys are no longer down, clear the group
        this.currentHotkey = null;
      }
    }
  }

  /**
   * Handle the actual change in position of associated object based on currently pressed keys. Called in step function
   * and keydown listener.
   *
   * @param delta - potential change in position in x and y for the position Property
   */
  private updatePosition( delta: number ): void {

    // hotkeys may disable dragging, so do this first
    this.updateHotkeys();

    if ( !this.hotkeyDisablingDragging ) {

      // handle the change in position
      let deltaX = 0;
      let deltaY = 0;

      if ( this.leftMovementKeysDown() ) {
        deltaX = -delta;
      }
      if ( this.rightMovementKeysDown() ) {
        deltaX = delta;
      }
      if ( this.upMovementKeysDown() ) {
        deltaY = -delta;
      }
      if ( this.downMovementKeysDown() ) {
        deltaY = delta;
      }

      // only initiate move if there was some attempted keyboard drag
      let vectorDelta = new Vector2( deltaX, deltaY );
      if ( !vectorDelta.equals( Vector2.ZERO ) ) {

        // to model coordinates
        if ( this._transform ) {
          vectorDelta = this._transform.inverseDelta2( vectorDelta );
        }

        // synchronize with model position
        if ( this._positionProperty ) {
          let newPosition = this._positionProperty.get().plus( vectorDelta );

          const dragBounds = this._dragBoundsProperty.value;

          // constrain to bounds in model coordinates
          if ( dragBounds ) {
            newPosition = dragBounds.closestPointTo( newPosition );
          }

          // update the position if it is different
          if ( !newPosition.equals( this._positionProperty.get() ) ) {
            this._positionProperty.set( newPosition );
          }
        }

        // call our drag function
        if ( this._drag ) {
          this._drag( vectorDelta );
        }

        this.dragEmitter.emit();
      }
    }
    this.moveOnHoldIntervalCounter = 0;
  }

  /**
   * Returns true if any of the keys in the list are currently down.
   */
  public keyInListDown( keys: string[] ): boolean {
    let keyIsDown = false;
    for ( let i = 0; i < this.keyState.length; i++ ) {
      if ( this.keyState[ i ].keyDown ) {
        for ( let j = 0; j < keys.length; j++ ) {
          if ( keys[ j ] === this.keyState[ i ].key ) {
            keyIsDown = true;
            break;
          }
        }
      }
      if ( keyIsDown ) {
        // no need to keep looking
        break;
      }
    }

    return keyIsDown;
  }

  /**
   * Return true if all keys in the list are currently held down.
   */
  public allKeysInListDown( keys: string[] ): boolean {
    assert && assert( keys.length > 0, 'You are testing to see if an empty list of keys is down?' );

    let allKeysDown = true;

    for ( let i = 0; i < keys.length; i++ ) {
      const foundKey = _.find( this.keyState, pressedKeyTiming => pressedKeyTiming.key === keys[ i ] );
      if ( !foundKey || !foundKey.keyDown ) {

        // key is not in the keystate or is not currently pressed down, all provided keys are not down
        allKeysDown = false;
        break;
      }
    }

    return allKeysDown;
  }

  /**
   * Returns true if the keystate indicates that a key is down that should move the object to the left.
   */
  public leftMovementKeysDown(): boolean {
    return this.keyInListDown( [ KeyboardUtils.KEY_A, KeyboardUtils.KEY_LEFT_ARROW ] );
  }

  /**
   * Returns true if the keystate indicates that a key is down that should move the object to the right.
   */
  public rightMovementKeysDown(): boolean {
    return this.keyInListDown( [ KeyboardUtils.KEY_RIGHT_ARROW, KeyboardUtils.KEY_D ] );
  }

  /**
   * Returns true if the keystate indicates that a key is down that should move the object up.
   */
  public upMovementKeysDown(): boolean {
    return this.keyInListDown( [ KeyboardUtils.KEY_UP_ARROW, KeyboardUtils.KEY_W ] );
  }

  /**
   * Returns true if the keystate indicates that a key is down that should move the upject down.
   */
  public downMovementKeysDown(): boolean {
    return this.keyInListDown( [ KeyboardUtils.KEY_DOWN_ARROW, KeyboardUtils.KEY_S ] );
  }

  /**
   * Returns true if any of the movement keys are down (arrow keys or WASD keys).
   */
  public getMovementKeysDown(): boolean {
    return this.rightMovementKeysDown() || this.leftMovementKeysDown() ||
           this.upMovementKeysDown() || this.downMovementKeysDown();
  }

  get movementKeysDown() { return this.getMovementKeysDown(); }

  /**
   * Returns true if the enter key is currently pressed down.
   */
  public enterKeyDown(): boolean {
    return this.keyInListDown( [ KeyboardUtils.KEY_ENTER ] );
  }

  /**
   * Returns true if the keystate indicates that the shift key is currently down.
   */
  public shiftKeyDown(): boolean {
    return this.keyInListDown( KeyboardUtils.SHIFT_KEYS );
  }

  /**
   * Add a hotkey that behaves such that the desired callback will be called when all keys listed in the array are
   * pressed down in order.
   */
  public addHotkey( hotkey: Hotkey ): void {
    this._hotkeys.push( hotkey );
  }

  /**
   * Remove a hotkey that was added with addHotkey.
   */
  public removeHotkey( hotkey: Hotkey ): void {
    assert && assert( this._hotkeys.includes( hotkey ), 'Trying to remove a hotkey that is not in the list of hotkeys.' );

    const hotkeyIndex = this._hotkeys.indexOf( hotkey );
    this._hotkeys.splice( hotkeyIndex, 1 );
  }

  /**
   * Sets the hotkeys of the KeyboardDragListener to passed-in array.
   */
  public setHotkeys( hotkeys: Hotkey[] ): void {
    this._hotkeys = hotkeys.slice( 0 ); // shallow copy
  }

  /**
   * See setHotkeys() for more information.
   */
  set hotkeys( hotkeys: Hotkey[] ) {
    this.setHotkeys( hotkeys );
  }

  /**
   * Clear all hotkeys from this KeyboardDragListener.
   */
  public removeAllHotkeys(): void {
    this._hotkeys = [];
  }

  /**
   * Resets the timers and control variables for the press and hold functionality.
   */
  private resetPressAndHold(): void {
    this.delayComplete = false;
    this.moveOnHoldDelayCounter = 0;
    this.moveOnHoldIntervalCounter = 0;
  }

  /**
   * Resets the timers and control variables for the hotkey functionality.
   */
  private resetHotkeyState(): void {
    this.currentHotkey = null;
    this.hotkeyHoldIntervalCounter = this._hotkeyHoldInterval; // reset to threshold so the hotkey fires immediately next time.
    this.hotkeyDisablingDragging = false;
  }

  /**
   * Reset the keystate Object tracking which keys are currently pressed down.
   */
  public interrupt(): void {
    this.keyState = [];
    this.resetHotkeyState();
    this.resetPressAndHold();

    if ( this._pointer ) {
      assert && assert( this._pointer.listeners.includes( this._pointerListener ),
        'A reference to the Pointer means it should have the pointerListener' );
      this._pointer!.removeInputListener( this._pointerListener );
      this._pointer = null;
    }
  }

  /**
   * Make eligible for garbage collection.
   */
  public override dispose(): void {
    this.interrupt();
    this._disposeKeyboardDragListener();
  }

  /**
   * Returns true if the key corresponds to a key that should move the object to the left.
   */
  public static isLeftMovementKey( key: string ): boolean {
    return key === KeyboardUtils.KEY_A || key === KeyboardUtils.KEY_LEFT_ARROW;
  }

  /**
   * Returns true if the key corresponds to a key that should move the object to the right.
   */
  public static isRightMovementKey( key: string ): boolean {
    return key === KeyboardUtils.KEY_D || key === KeyboardUtils.KEY_RIGHT_ARROW;
  }

  /**
   * Returns true if the key corresponds to a key that should move the object up.
   */
  private static isUpMovementKey( key: string ): boolean {
    return key === KeyboardUtils.KEY_W || key === KeyboardUtils.KEY_UP_ARROW;
  }

  /**
   * Returns true if the key corresponds to a key that should move the object down.
   */
  public static isDownMovementKey( key: string ): boolean {
    return key === KeyboardUtils.KEY_S || key === KeyboardUtils.KEY_DOWN_ARROW;
  }
}

scenery.register( 'KeyboardDragListener', KeyboardDragListener );

export default KeyboardDragListener;