// Copyright 2019-2023, University of Colorado Boulder

/**
 * A general type for keyboard dragging. Objects can be dragged in one or two dimensions with the arrow keys and with
 * the WASD keys. See the option keyboardDragDirection for a description of how keyboard keys can be mapped to
 * motion for 1D and 2D motion. This can be added to a node through addInputListener for accessibility, which is mixed
 * into Nodes with the ParallelDOM trait.
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
import { KeyboardUtils, Node, PDOMPointer, scenery, SceneryEvent, TInputListener } from '../imports.js';
import TProperty from '../../../axon/js/TProperty.js';
import optionize from '../../../phet-core/js/optionize.js';
import TReadOnlyProperty from '../../../axon/js/TReadOnlyProperty.js';
import TEmitter from '../../../axon/js/TEmitter.js';
import assertMutuallyExclusiveOptions from '../../../phet-core/js/assertMutuallyExclusiveOptions.js';

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

// Possible movement types for this KeyboardDragListener. 2D motion ('both') or 1D motion ('leftRight' or 'upDown').
type KeyboardDragDirection = 'both' | 'leftRight' | 'upDown';

type KeyboardDragDirectionKeys = {
  left: string[];
  right: string[];
  up: string[];
  down: string[];
};

const KEYBOARD_DRAG_DIRECTION_KEY_MAP = new Map<KeyboardDragDirection, KeyboardDragDirectionKeys>( [
  [ 'both', {
    left: [ KeyboardUtils.KEY_A, KeyboardUtils.KEY_LEFT_ARROW ],
    right: [ KeyboardUtils.KEY_RIGHT_ARROW, KeyboardUtils.KEY_D ],
    up: [ KeyboardUtils.KEY_UP_ARROW, KeyboardUtils.KEY_W ],
    down: [ KeyboardUtils.KEY_DOWN_ARROW, KeyboardUtils.KEY_S ]
  } ],
  [ 'leftRight', {
    left: [ KeyboardUtils.KEY_A, KeyboardUtils.KEY_LEFT_ARROW, KeyboardUtils.KEY_DOWN_ARROW, KeyboardUtils.KEY_S ],
    right: [ KeyboardUtils.KEY_RIGHT_ARROW, KeyboardUtils.KEY_D, KeyboardUtils.KEY_UP_ARROW, KeyboardUtils.KEY_W ],
    up: [],
    down: []
  } ],
  [ 'upDown', {
    left: [],
    right: [],
    up: [ KeyboardUtils.KEY_RIGHT_ARROW, KeyboardUtils.KEY_D, KeyboardUtils.KEY_UP_ARROW, KeyboardUtils.KEY_W ],
    down: [ KeyboardUtils.KEY_A, KeyboardUtils.KEY_LEFT_ARROW, KeyboardUtils.KEY_DOWN_ARROW, KeyboardUtils.KEY_S ]
  } ]
] );

type MapPosition = ( point: Vector2 ) => Vector2;

type SelfOptions = {

  // How much the position Property will change in view coordinates every moveOnHoldInterval. Object will
  // move in discrete steps at this interval. If you would like smoother "animated" motion use dragVelocity
  // instead. dragDelta produces a UX that is more typical for applications but dragVelocity is better for video
  // game-like components. dragDelta and dragVelocity are mutually exclusive options.
  dragDelta?: number;

  // How much the PositionProperty will change in view coordinates every moveOnHoldInterval while the shift modifier
  // key is pressed. Shift modifier should produce more fine-grained motion so this value needs to be less than
  // dragDelta if provided. Object will move in discrete steps. If you would like smoother "animated" motion use
  // dragVelocity options instead. dragDelta options produce a UX that is more typical for applications but dragVelocity
  // is better for game-like components. dragDelta and dragVelocity are mutually exclusive options.
  shiftDragDelta?: number;

  // While a direction key is held down, the target will move by this amount in view coordinates every second.
  // This is an alternative way to control motion with keyboard than dragDelta and produces smoother motion for
  // the object. dragVelocity and dragDelta options are mutually exclusive. See dragDelta for more information.
  dragVelocity?: number;

  // While a direction key is held down with the shift modifier key, the target will move by this amount in view
  // coordinates every second. Shift modifier should produce more fine-grained motion so this value needs to be less
  // than dragVelocity if provided. This is an alternative way to control motion with keyboard than dragDelta and
  // produces smoother motion for the object. dragVelocity and dragDelta options are mutually exclusive. See dragDelta
  // for more information.
  shiftDragVelocity?: number;

  // Specifies the direction of motion for the KeyboardDragListener. By default, the position Vector2 can change in
  // both directions by pressing the arrow keys. But you can constrain dragging to 1D left-right or up-down motion
  // with this value.
  keyboardDragDirection?: KeyboardDragDirection;

  // If provided, it will be synchronized with the drag position in the model frame, applying provided transforms as
  // needed. Most useful when used with transform option
  positionProperty?: TProperty<Vector2> | null;

  // If provided, this will be the conversion between the view and model coordinate frames. Usually most useful when
  // paired with the positionProperty.
  transform?: Transform3 | TReadOnlyProperty<Transform3> | null;

  // If provided, the model position will be constrained to be inside these bounds, in model coordinates
  dragBoundsProperty?: TReadOnlyProperty<Bounds2 | null> | null;

  // If provided, it will allow custom mapping
  // from the desired position (i.e. where the pointer is) to the actual possible position (i.e. where the dragged
  // object ends up). For example, using dragBoundsProperty is equivalent to passing:
  //   mapPosition: function( point ) { return dragBoundsProperty.value.closestPointTo( point ); }
  mapPosition?: MapPosition | null;

  // Called when keyboard drag is started (on initial press).
  start?: ( ( event: SceneryEvent ) => void ) | null;

  // Called during drag. Note that this does not provide the SceneryEvent. Dragging happens during animation
  // (as long as keys are down), so there is no event associated with the drag.
  drag?: ( ( viewDelta: Vector2 ) => void ) | null;

  // Called when keyboard dragging ends.
  end?: ( ( event?: SceneryEvent ) => void ) | null;

  // Arrow keys must be pressed this long to begin movement set on moveOnHoldInterval, in ms
  moveOnHoldDelay?: number;

  // Time interval at which the object will change position while the arrow key is held down, in ms. This must be larger
  // than 0 to prevent dragging that is based on how often animation-frame steps occur.
  moveOnHoldInterval?: number;

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

class KeyboardDragListener extends EnabledComponent implements TInputListener {

  // See options for documentation
  private _start: ( ( event: SceneryEvent ) => void ) | null;
  private _drag: ( ( viewDelta: Vector2, listener: KeyboardDragListener ) => void ) | null;
  private _end: ( ( event?: SceneryEvent ) => void ) | null;
  private _dragBoundsProperty: TReadOnlyProperty<Bounds2 | null>;
  private _mapPosition: MapPosition | null;
  private _transform: Transform3 | TReadOnlyProperty<Transform3> | null;
  private _keyboardDragDirection: KeyboardDragDirection;
  private _positionProperty: TProperty<Vector2> | null;
  private _dragVelocity: number;
  private _shiftDragVelocity: number;
  private _dragDelta: number;
  private _shiftDragDelta: number;
  private _moveOnHoldDelay: number;
  private _moveOnHoldInterval!: number;
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
  public dragEmitter: TEmitter;

  // Implements disposal
  private readonly _disposeKeyboardDragListener: () => void;

  // A listener added to the pointer when dragging starts so that we can attach a listener and provide a channel of
  // communication to the AnimatedPanZoomListener to define custom behavior for screen panning during a drag operation.
  private readonly _pointerListener: TInputListener;

  // A reference to the Pointer during a drag operation so that we can add/remove the _pointerListener.
  private _pointer: PDOMPointer | null;

  // Whether we are using a velocity implementation or delta implementation for dragging. See options
  // dragDelta and dragVelocity for more information.
  private readonly useDragVelocity: boolean;

  public constructor( providedOptions?: KeyboardDragListenerOptions ) {

    // Use either dragVelocity or dragDelta, cannot use both at the same time.
    assert && assertMutuallyExclusiveOptions( providedOptions, [ 'dragVelocity', 'shiftDragVelocity' ], [ 'dragDelta', 'shiftDragDelta' ] );
    assert && assertMutuallyExclusiveOptions( providedOptions, [ 'mapPosition' ], [ 'dragBoundsProperty' ] );

    const options = optionize<KeyboardDragListenerOptions, SelfOptions, EnabledComponentOptions>()( {

      // default moves the object roughly 600 view coordinates every second, assuming 60 fps
      dragDelta: 10,
      shiftDragDelta: 5,
      dragVelocity: 0,
      shiftDragVelocity: 0,
      keyboardDragDirection: 'both',
      positionProperty: null,
      transform: null,
      dragBoundsProperty: null,
      mapPosition: null,
      start: null,
      drag: null,
      end: null,
      moveOnHoldDelay: 0,
      moveOnHoldInterval: 1000 / 60, // an average dt value at 60 frames a second
      hotkeyHoldInterval: 800,
      phetioEnabledPropertyInstrumented: false,
      tandem: Tandem.REQUIRED,

      // DragListener by default doesn't allow PhET-iO to trigger drag Action events
      phetioReadOnly: true
    }, providedOptions );

    assert && assert( options.shiftDragVelocity <= options.dragVelocity, 'shiftDragVelocity should be less than or equal to shiftDragVelocity, it is intended to provide more fine-grained control' );
    assert && assert( options.shiftDragDelta <= options.dragDelta, 'shiftDragDelta should be less than or equal to dragDelta, it is intended to provide more fine-grained control' );

    super( options );

    // mutable attributes declared from options, see options for info, as well as getters and setters
    this._start = options.start;
    this._drag = options.drag;
    this._end = options.end;
    this._dragBoundsProperty = ( options.dragBoundsProperty || new Property( null ) );
    this._mapPosition = options.mapPosition;
    this._transform = options.transform;
    this._positionProperty = options.positionProperty;
    this._dragVelocity = options.dragVelocity;
    this._shiftDragVelocity = options.shiftDragVelocity;
    this._dragDelta = options.dragDelta;
    this._shiftDragDelta = options.shiftDragDelta;
    this._moveOnHoldDelay = options.moveOnHoldDelay;
    this.moveOnHoldInterval = options.moveOnHoldInterval;
    this._hotkeyHoldInterval = options.hotkeyHoldInterval;
    this._keyboardDragDirection = options.keyboardDragDirection;

    this.keyState = [];
    this._hotkeys = [];
    this.currentHotkey = null;
    this.hotkeyDisablingDragging = false;

    // This is initialized to the "threshold" so that the first hotkey will fire immediately. Only subsequent actions
    // while holding the hotkey should result in a delay of this much. in ms
    this.hotkeyHoldIntervalCounter = this._hotkeyHoldInterval;

    // for readability - since dragVelocity and dragDelta are mutually exclusive, a value for either one of these
    // indicates dragging implementation should use velocity
    this.useDragVelocity = options.dragVelocity > 0 || options.shiftDragVelocity > 0;

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

      // initial movement on down should only be used for dragDelta implementation
      if ( !this.useDragVelocity ) {

        // move object on first down before a delay
        const positionDelta = this.shiftKeyDown() ? this._shiftDragDelta : this._dragDelta;
        this.updatePosition( positionDelta );
        this.moveOnHoldIntervalCounter = 0;
      }
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

  public get dragBounds(): Bounds2 | null { return this.getDragBounds(); }

  /**
   * Sets the drag transform of the listener.
   */
  public setTransform( transform: Transform3 | TReadOnlyProperty<Transform3> | null ): void {
    this._transform = transform;
  }

  public set transform( transform: Transform3 | TReadOnlyProperty<Transform3> | null ) { this.setTransform( transform ); }

  public get transform(): Transform3 | TReadOnlyProperty<Transform3> | null { return this.getTransform(); }

  /**
   * Returns the transform of the listener.
   */
  public getTransform(): Transform3 | TReadOnlyProperty<Transform3> | null {
    return this._transform;
  }

  /**
   * Getter for the dragVelocity property, see options.dragVelocity for more info.
   */
  public get dragVelocity(): number { return this._dragVelocity; }

  /**
   * Setter for the dragVelocity property, see options.dragVelocity for more info.
   */
  public set dragVelocity( dragVelocity: number ) { this._dragVelocity = dragVelocity; }

  /**
   * Getter for the shiftDragVelocity property, see options.shiftDragVelocity for more info.
   */
  public get shiftDragVelocity(): number { return this._shiftDragVelocity; }

  /**
   * Setter for the shiftDragVelocity property, see options.shiftDragVelocity for more info.
   */
  public set shiftDragVelocity( shiftDragVelocity: number ) { this._shiftDragVelocity = shiftDragVelocity; }

  /**
   * Getter for the dragDelta property, see options.dragDelta for more info.
   */
  public get dragDelta(): number { return this._dragDelta; }

  /**
   * Setter for the dragDelta property, see options.dragDelta for more info.
   */
  public set dragDelta( dragDelta: number ) { this._dragDelta = dragDelta; }

  /**
   * Getter for the shiftDragDelta property, see options.shiftDragDelta for more info.
   */
  public get shiftDragDelta(): number { return this._shiftDragDelta; }

  /**
   * Setter for the shiftDragDelta property, see options.shiftDragDelta for more info.
   */
  public set shiftDragDelta( shiftDragDelta: number ) { this._shiftDragDelta = shiftDragDelta; }

  /**
   * Getter for the moveOnHoldDelay property, see options.moveOnHoldDelay for more info.
   */
  public get moveOnHoldDelay(): number { return this._moveOnHoldDelay; }

  /**
   * Setter for the moveOnHoldDelay property, see options.moveOnHoldDelay for more info.
   */
  public set moveOnHoldDelay( moveOnHoldDelay: number ) { this._moveOnHoldDelay = moveOnHoldDelay; }

  /**
   * Getter for the moveOnHoldInterval property, see options.moveOnHoldInterval for more info.
   */
  public get moveOnHoldInterval(): number { return this._moveOnHoldInterval; }

  /**
   * Setter for the moveOnHoldInterval property, see options.moveOnHoldInterval for more info.
   */
  public set moveOnHoldInterval( moveOnHoldInterval: number ) {
    assert && assert( moveOnHoldInterval > 0, 'if the moveOnHoldInterval is 0, then the dragging will be ' +
                                              'dependent on how often the dragListener is stepped' );
    this._moveOnHoldInterval = moveOnHoldInterval;
  }

  /**
   * Getter for the hotkeyHoldInterval property, see options.hotkeyHoldInterval for more info.
   */
  public get hotkeyHoldInterval(): number { return this._hotkeyHoldInterval; }

  /**
   * Setter for the hotkeyHoldInterval property, see options.hotkeyHoldInterval for more info.
   */
  public set hotkeyHoldInterval( hotkeyHoldInterval: number ) { this._hotkeyHoldInterval = hotkeyHoldInterval; }

  public get isPressed(): boolean {
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
   *
   * focusout bubbles, which is important so that the work of interrupt happens as focus moves between children of
   * a parent with a KeyboardDragListener, which can create state for the keystate.
   * See https://github.com/phetsims/scenery/issues/1461.
   */
  public focusout( event: SceneryEvent ): void {
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

    // dt is in seconds and we convert to ms
    const ms = dt * 1000;

    // no-op unless a key is down
    if ( this.keyState.length > 0 ) {
      // for each key that is still down, increment the tracked time that has been down
      for ( let i = 0; i < this.keyState.length; i++ ) {
        if ( this.keyState[ i ].keyDown ) {
          this.keyState[ i ].timeDown += ms;
        }
      }

      // Movement delay counters should only increment if movement keys are pressed down. They will get reset
      // every up event.
      if ( this.movementKeysDown ) {
        this.moveOnHoldDelayCounter += ms;
        this.moveOnHoldIntervalCounter += ms;
      }

      // update timer for keygroup if one is being held down
      if ( this.currentHotkey ) {
        this.hotkeyHoldIntervalCounter += ms;
      }

      let positionDelta = 0;

      if ( this.useDragVelocity ) {

        // calculate change in position from time step
        const positionVelocitySeconds = this.shiftKeyDown() ? this._shiftDragVelocity : this._dragVelocity;
        const positionVelocityMilliseconds = positionVelocitySeconds / 1000;
        positionDelta = ms * positionVelocityMilliseconds;
      }
      else {

        // If dragging by deltas, we are only movable every moveOnHoldInterval.
        let movable = false;

        // Wait for a longer delay (moveOnHoldDelay) on initial press and hold.
        if ( this.moveOnHoldDelayCounter >= this._moveOnHoldDelay && !this.delayComplete ) {
          movable = true;
          this.delayComplete = true;
          this.moveOnHoldIntervalCounter = 0;
        }

        // Initial delay is complete, now we will move every moveOnHoldInterval
        if ( this.delayComplete && this.moveOnHoldIntervalCounter >= this._moveOnHoldInterval ) {
          movable = true;

          // If updating as a result of the moveOnHoldIntervalCounter, don't automatically throw away any "remainder"
          // time by setting back to 0. We want to accumulate them so that, no matter the clock speed of the
          // runtime, the long-term effect of the drag is consistent.
          const overflowTime = this.moveOnHoldIntervalCounter - this._moveOnHoldInterval; // ms

          // This doesn't take into account if 2 updatePosition calls should occur based on the current timing.
          this.moveOnHoldIntervalCounter = overflowTime;
        }

        positionDelta = movable ? ( this.shiftKeyDown() ? this._shiftDragDelta : this._dragDelta ) : 0;
      }

      if ( positionDelta > 0 ) {
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
        deltaX -= delta;
      }
      if ( this.rightMovementKeysDown() ) {
        deltaX += delta;
      }

      if ( this.upMovementKeysDown() ) {
        deltaY -= delta;
      }
      if ( this.downMovementKeysDown() ) {
        deltaY += delta;
      }

      // only initiate move if there was some attempted keyboard drag
      let vectorDelta = new Vector2( deltaX, deltaY );
      if ( !vectorDelta.equals( Vector2.ZERO ) ) {

        // to model coordinates
        if ( this._transform ) {
          const transform = this._transform instanceof Transform3 ? this._transform : this._transform.value;

          vectorDelta = transform.inverseDelta2( vectorDelta );
        }

        // synchronize with model position
        if ( this._positionProperty ) {
          let newPosition = this._positionProperty.get().plus( vectorDelta );

          newPosition = this.mapModelPoint( newPosition );

          // update the position if it is different
          if ( !newPosition.equals( this._positionProperty.get() ) ) {
            this._positionProperty.set( newPosition );
          }
        }

        // call our drag function
        if ( this._drag ) {
          this._drag( vectorDelta, this );
        }

        this.dragEmitter.emit();
      }
    }
  }

  /**
   * Apply a mapping from the drag target's model position to an allowed model position.
   *
   * A common example is using dragBounds, where the position of the drag target is constrained to within a bounding
   * box. This is done by mapping points outside the bounding box to the closest position inside the box. More
   * general mappings can be used.
   *
   * Should be overridden (or use mapPosition) if a custom transformation is needed.
   *
   * @returns - A point in the model coordinate frame
   */
  protected mapModelPoint( modelPoint: Vector2 ): Vector2 {
    if ( this._mapPosition ) {
      return this._mapPosition( modelPoint );
    }
    else if ( this._dragBoundsProperty.value ) {
      return this._dragBoundsProperty.value.closestPointTo( modelPoint );
    }
    else {
      return modelPoint;
    }
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
   * Get the keyboard keys for the KeyboardDragDirection of this KeyboardDragListener.
   */
  private getKeyboardDragDirectionKeys(): KeyboardDragDirectionKeys {
    const directionKeys = KEYBOARD_DRAG_DIRECTION_KEY_MAP.get( this._keyboardDragDirection )!;
    assert && assert( directionKeys, `No direction keys found in map for KeyboardDragDirection ${this._keyboardDragDirection}` );
    return directionKeys;
  }

  /**
   * Returns true if the keystate indicates that a key is down that should move the object to the left.
   */
  public leftMovementKeysDown(): boolean {
    return this.keyInListDown( this.getKeyboardDragDirectionKeys().left );
  }

  /**
   * Returns true if the keystate indicates that a key is down that should move the object to the right.
   */
  public rightMovementKeysDown(): boolean {
    return this.keyInListDown( this.getKeyboardDragDirectionKeys().right );
  }

  /**
   * Returns true if the keystate indicates that a key is down that should move the object up.
   */
  public upMovementKeysDown(): boolean {
    return this.keyInListDown( this.getKeyboardDragDirectionKeys().up );
  }

  /**
   * Returns true if the keystate indicates that a key is down that should move the upject down.
   */
  public downMovementKeysDown(): boolean {
    return this.keyInListDown( this.getKeyboardDragDirectionKeys().down );
  }

  /**
   * Returns true if any of the movement keys are down (arrow keys or WASD keys).
   */
  public getMovementKeysDown(): boolean {
    return this.rightMovementKeysDown() || this.leftMovementKeysDown() ||
           this.upMovementKeysDown() || this.downMovementKeysDown();
  }

  public get movementKeysDown(): boolean { return this.getMovementKeysDown(); }

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
  public set hotkeys( hotkeys: Hotkey[] ) {
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
      this._pointer.removeInputListener( this._pointerListener );
      this._pointer = null;

      this._end && this._end();
    }
  }

  /**
   * Make eligible for garbage collection.
   */
  public override dispose(): void {
    this.interrupt();
    this._disposeKeyboardDragListener();
    super.dispose();
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