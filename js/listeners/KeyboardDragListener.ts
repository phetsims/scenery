// Copyright 2019-2024, University of Colorado Boulder

/**
 * An input listener for keyboard-based drag interactions, allowing objects to be moved using the arrow keys or
 * the W, A, S, D keys.
 *
 * Key features:
 * - Supports both discrete (step-based) and continuous (speed-based) dragging modes.
 * - Allows restricting movement to specific axes (e.g., horizontal or vertical only) or allowing free 2D movement.
 * - Configurable drag speed and drag delta values, with separate configurations when the shift key is held for
 *   finer control.
 * - Optionally synchronizes with a 'positionProperty' to allow for model-view coordination with custom transformations
 *   if needed.
 * - Provides hooks for start, drag (movement), and end phases of a drag interaction through callback options.
 * - Includes support for drag bounds, restricting the draggable area within specified model coordinates.
 * - Utilizes a CallbackTimer for smooth, timed updates during drag operations, especially useful in continuous drag
 *   mode.
 *
 * Usage:
 * Attach an instance of KeyboardDragListener to a Node via the `addInputListener` method.
 *
 * Example:
 *
 *   const myNode = new Node();
 *   const dragListener = new KeyboardDragListener( {
 *     dragDelta: 2,
 *     shiftDragDelta: 2,
 *     start: (event, listener) => { console.log('Drag started'); },
 *     drag: (event, listener) => { console.log('Dragging'); },
 *     end: (event, listener) => { console.log('Drag ended'); },
 *     positionProperty: myNode.positionProperty,
 *     transform: myNode.getTransform()
 *   } );
 *   myNode.addInputListener(dragListener);
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 * @author Michael Barlow
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */

import PhetioAction from '../../../tandem/js/PhetioAction.js';
import Property from '../../../axon/js/Property.js';
import Bounds2 from '../../../dot/js/Bounds2.js';
import Transform3 from '../../../dot/js/Transform3.js';
import Vector2 from '../../../dot/js/Vector2.js';
import EventType from '../../../tandem/js/EventType.js';
import Tandem from '../../../tandem/js/Tandem.js';
import { globalKeyStateTracker, KeyboardListener, KeyboardListenerOptions, KeyboardUtils, Node, PDOMPointer, scenery, SceneryEvent, TInputListener } from '../imports.js';
import TProperty from '../../../axon/js/TProperty.js';
import optionize, { EmptySelfOptions } from '../../../phet-core/js/optionize.js';
import TReadOnlyProperty from '../../../axon/js/TReadOnlyProperty.js';
import assertMutuallyExclusiveOptions from '../../../phet-core/js/assertMutuallyExclusiveOptions.js';
import { PhetioObjectOptions } from '../../../tandem/js/PhetioObject.js';
import TinyProperty from '../../../axon/js/TinyProperty.js';
import CallbackTimer from '../../../axon/js/CallbackTimer.js';
import PickOptional from '../../../phet-core/js/types/PickOptional.js';
import platform from '../../../phet-core/js/platform.js';
import StrictOmit from '../../../phet-core/js/types/StrictOmit.js';
import { EnabledComponentOptions } from '../../../axon/js/EnabledComponent.js';

// 'shift' is not included in any list of keys because we don't want the KeyboardListener to be 'pressed' when only
// the shift key is down. State of the shift key is tracked by the globalKeyStateTracker.
const allKeys = [ 'arrowLeft', 'arrowRight', 'arrowUp', 'arrowDown', 'w', 'a', 's', 'd' ] as const;
const leftRightKeys = [ 'arrowLeft', 'arrowRight', 'a', 'd' ] as const;
const upDownKeys = [ 'arrowUp', 'arrowDown', 'w', 's' ] as const;

type KeyboardDragListenerKeyStroke = typeof allKeys | typeof leftRightKeys | typeof upDownKeys;

// Possible movement types for this KeyboardDragListener. 2D motion ('both') or 1D motion ('leftRight' or 'upDown').
type KeyboardDragDirection = 'both' | 'leftRight' | 'upDown';
const KeyboardDragDirectionToKeysMap = new Map<KeyboardDragDirection, KeyboardDragListenerKeyStroke>( [
  [ 'both', allKeys ],
  [ 'leftRight', leftRightKeys ],
  [ 'upDown', upDownKeys ]
] );

type MapPosition = ( point: Vector2 ) => Vector2;

type SelfOptions = {

  // How much the position Property will change in view coordinates every moveOnHoldInterval. Object will
  // move in discrete steps at this interval. If you would like smoother "animated" motion use dragSpeed
  // instead. dragDelta produces a UX that is more typical for applications but dragSpeed is better for video
  // game-like components. dragDelta and dragSpeed are mutually exclusive options.
  dragDelta?: number;

  // How much the PositionProperty will change in view coordinates every moveOnHoldInterval while the shift modifier
  // key is pressed. Shift modifier should produce more fine-grained motion so this value needs to be less than
  // dragDelta if provided. Object will move in discrete steps. If you would like smoother "animated" motion use
  // dragSpeed options instead. dragDelta options produce a UX that is more typical for applications but dragSpeed
  // is better for game-like components. dragDelta and dragSpeed are mutually exclusive options.
  shiftDragDelta?: number;

  // While a direction key is held down, the target will move by this amount in view coordinates every second.
  // This is an alternative way to control motion with keyboard than dragDelta and produces smoother motion for
  // the object. dragSpeed and dragDelta options are mutually exclusive. See dragDelta for more information.
  dragSpeed?: number;

  // While a direction key is held down with the shift modifier key, the target will move by this amount in view
  // coordinates every second. Shift modifier should produce more fine-grained motion so this value needs to be less
  // than dragSpeed if provided. This is an alternative way to control motion with keyboard than dragDelta and
  // produces smoother motion for the object. dragSpeed and dragDelta options are mutually exclusive. See dragDelta
  // for more information.
  shiftDragSpeed?: number;

  // Specifies the direction of motion for the KeyboardDragListener. By default, the position Vector2 can change in
  // both directions by pressing the arrow keys. But you can constrain dragging to 1D left-right or up-down motion
  // with this value.
  keyboardDragDirection?: KeyboardDragDirection;

  // If provided, it will be synchronized with the drag position in the model frame, applying provided transforms as
  // needed. Most useful when used with transform option
  positionProperty?: Pick<TProperty<Vector2>, 'value'> | null;

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
  start?: ( ( event: SceneryEvent, listener: KeyboardDragListener ) => void ) | null;

  // Called during drag. If providedOptions.transform is provided, vectorDelta will be in model coordinates.
  // Otherwise, it will be in view coordinates. Note that this does not provide the SceneryEvent. Dragging
  // happens during animation (as long as keys are down), so there is no event associated with the drag.
  drag?: ( ( event: SceneryEvent, listener: KeyboardDragListener ) => void ) | null;

  // Called when keyboard dragging ends.
  end?: ( ( event: SceneryEvent | null, listener: KeyboardDragListener ) => void ) | null;

  // Arrow keys must be pressed this long to begin movement set on moveOnHoldInterval, in ms
  moveOnHoldDelay?: number;

  // Time interval at which the object will change position while the arrow key is held down, in ms. This must be larger
  // than 0 to prevent dragging that is based on how often animation-frame steps occur.
  moveOnHoldInterval?: number;

  // Though DragListener is not instrumented, declare these here to support properly passing this to children, see
  // https://github.com/phetsims/tandem/issues/60.
} & Pick<PhetioObjectOptions, 'tandem' | 'phetioReadOnly'>;

type ParentOptions = StrictOmit<KeyboardListenerOptions<KeyboardDragListenerKeyStroke>, 'keys'>;

export type KeyboardDragListenerOptions = SelfOptions & // Options specific to this class
  PickOptional<ParentOptions, 'focus' | 'blur'> & // Only focus/blur are optional from the superclass
  EnabledComponentOptions; // Other superclass options are allowed

class KeyboardDragListener extends KeyboardListener<KeyboardDragListenerKeyStroke> {

  // See options for documentation
  private readonly _start: ( ( event: SceneryEvent, listener: KeyboardDragListener ) => void ) | null;
  private readonly _drag: ( ( event: SceneryEvent, listener: KeyboardDragListener ) => void ) | null;
  private readonly _end: ( ( event: SceneryEvent | null, listener: KeyboardDragListener ) => void ) | null;
  private _dragBoundsProperty: TReadOnlyProperty<Bounds2 | null>;
  private readonly _mapPosition: MapPosition | null;
  private _transform: Transform3 | TReadOnlyProperty<Transform3> | null;
  private readonly _positionProperty: Pick<TProperty<Vector2>, 'value'> | null;
  private _dragSpeed: number;
  private _shiftDragSpeed: number;
  private _dragDelta: number;
  private _shiftDragDelta: number;
  private readonly _moveOnHoldDelay: number;

  // Properties internal to the listener that track pressed keys. Instead of updating in the KeyboardListener
  // callback, the positionProperty is updated in a callback timer depending on the state of these Properties
  // so that movement is smooth.
  private leftKeyDownProperty = new TinyProperty<boolean>( false );
  private rightKeyDownProperty = new TinyProperty<boolean>( false );
  private upKeyDownProperty = new TinyProperty<boolean>( false );
  private downKeyDownProperty = new TinyProperty<boolean>( false );

  // Fires to conduct the start and end of a drag, added for PhET-iO interoperability
  private dragStartAction: PhetioAction<[ SceneryEvent ]>;
  private dragEndAction: PhetioAction;
  private dragAction: PhetioAction;

  // KeyboardDragListener is implemented with KeyboardListener and therefore Hotkey. Hotkeys use 'global' DOM events
  // instead of SceneryEvent dispatch. In order to start the drag with a SceneryEvent, this listener waits
  // to start until its keys are pressed, and it starts the drag on the next SceneryEvent from keydown dispatch.
  private startNextKeyboardEvent = false;

  // Similar to the above, but used for restarting the callback timer on the next keydown event when a new key is
  // pressed.
  private restartTimerNextKeyboardEvent = false;

  // Implements disposal.
  private readonly _disposeKeyboardDragListener: () => void;

  // A listener added to the pointer when dragging starts so that we can attach a listener and provide a channel of
  // communication to the AnimatedPanZoomListener to define custom behavior for screen panning during a drag operation.
  private readonly _pointerListener: TInputListener;

  // A reference to the Pointer during a drag operation so that we can add/remove the _pointerListener.
  private _pointer: PDOMPointer | null;

  // Whether this listener uses a speed implementation or delta implementation for dragging. See options
  // dragSpeed and dragDelta for more information.
  private readonly useDragSpeed: boolean;

  // The vector delta that is used to move the object during a drag operation. Assigned to the listener so that
  // it is usable in the drag callback.
  public vectorDelta: Vector2 = new Vector2( 0, 0 );

  // The callback timer that is used to move the object during a drag operation to support animated motion and
  // motion every moveOnHoldInterval.
  private readonly callbackTimer: CallbackTimer;

  public constructor( providedOptions?: KeyboardDragListenerOptions ) {

    // Use either dragSpeed or dragDelta, cannot use both at the same time.
    assert && assertMutuallyExclusiveOptions( providedOptions, [ 'dragSpeed', 'shiftDragSpeed' ], [ 'dragDelta', 'shiftDragDelta' ] );

    // 'move on hold' timings are only relevant for 'delta' implementations of dragging
    assert && assertMutuallyExclusiveOptions( providedOptions, [ 'dragSpeed' ], [ 'moveOnHoldDelay', 'moveOnHOldInterval' ] );
    assert && assertMutuallyExclusiveOptions( providedOptions, [ 'mapPosition' ], [ 'dragBoundsProperty' ] );

    const options = optionize<KeyboardDragListenerOptions, SelfOptions, ParentOptions>()( {

      // default moves the object roughly 600 view coordinates every second, assuming 60 fps
      dragDelta: 10,
      shiftDragDelta: 5,
      dragSpeed: 0,
      shiftDragSpeed: 0,
      keyboardDragDirection: 'both',
      positionProperty: null,
      transform: null,
      dragBoundsProperty: null,
      mapPosition: null,
      start: null,
      drag: null,
      end: null,
      moveOnHoldDelay: 500,
      moveOnHoldInterval: 400,
      tandem: Tandem.REQUIRED,

      // DragListener by default doesn't allow PhET-iO to trigger drag Action events
      phetioReadOnly: true
    }, providedOptions );

    assert && assert( options.shiftDragSpeed <= options.dragSpeed, 'shiftDragSpeed should be less than or equal to shiftDragSpeed, it is intended to provide more fine-grained control' );
    assert && assert( options.shiftDragDelta <= options.dragDelta, 'shiftDragDelta should be less than or equal to dragDelta, it is intended to provide more fine-grained control' );

    const keys = KeyboardDragDirectionToKeysMap.get( options.keyboardDragDirection )!;
    assert && assert( keys, 'Invalid keyboardDragDirection' );

    const superOptions = optionize<KeyboardDragListenerOptions, EmptySelfOptions, KeyboardListenerOptions<KeyboardDragListenerKeyStroke>>()( {
      keys: keys,

      // We still want to start drag operations when the shift modifier key is pressed, even though it is not
      // listed in keys.
      ignoredModifierKeys: [ 'shift' ]
    }, options );

    super( superOptions );

    // pressedKeysProperty comes from KeyboardListener, and it is used to determine the state of the movement keys.
    // This approach gives more control over the positionProperty in the callbackTimer than using the KeyboardListener
    // callback.
    this.pressedKeysProperty.link( pressedKeys => {
      this.leftKeyDownProperty.value = pressedKeys.includes( 'arrowLeft' ) || pressedKeys.includes( 'a' );
      this.rightKeyDownProperty.value = pressedKeys.includes( 'arrowRight' ) || pressedKeys.includes( 'd' );
      this.upKeyDownProperty.value = pressedKeys.includes( 'arrowUp' ) || pressedKeys.includes( 'w' );
      this.downKeyDownProperty.value = pressedKeys.includes( 'arrowDown' ) || pressedKeys.includes( 's' );
    } );

    // Mutable attributes declared from options, see options for info, as well as getters and setters.
    this._start = options.start;
    this._drag = options.drag;
    this._end = options.end;
    this._dragBoundsProperty = ( options.dragBoundsProperty || new Property( null ) );
    this._mapPosition = options.mapPosition;
    this._transform = options.transform;
    this._positionProperty = options.positionProperty;
    this._dragSpeed = options.dragSpeed;
    this._shiftDragSpeed = options.shiftDragSpeed;
    this._dragDelta = options.dragDelta;
    this._shiftDragDelta = options.shiftDragDelta;
    this._moveOnHoldDelay = options.moveOnHoldDelay;

    // Since dragSpeed and dragDelta are mutually-exclusive drag implementations, a value for either one of these
    // options indicates we should use a speed implementation for dragging.
    this.useDragSpeed = options.dragSpeed > 0 || options.shiftDragSpeed > 0;

    this.dragStartAction = new PhetioAction( event => {
      this._start && this._start( event, this );

      if ( this.useDragSpeed ) {
        this.callbackTimer.start();
      }
    }, {
      parameters: [ { name: 'event', phetioType: SceneryEvent.SceneryEventIO } ],
      tandem: options.tandem.createTandem( 'dragStartAction' ),
      phetioDocumentation: 'Emits whenever a keyboard drag starts.',
      phetioReadOnly: options.phetioReadOnly,
      phetioEventType: EventType.USER
    } );

    // The drag action only executes when there is actual movement (vectorDelta is non-zero). For example, it does
    // NOT execute if conflicting keys are pressed (e.g. left and right arrow keys at the same time). Note that this
    // is expected to be executed from the CallbackTimer. So there will be problems if this can be executed from
    // PhET-iO clients.
    this.dragAction = new PhetioAction( () => {
      assert && assert( this.isPressedProperty.value, 'The listener should not be dragging if not pressed' );

      // synchronize with model position
      if ( this._positionProperty ) {
        let newPosition = this._positionProperty.value.plus( this.vectorDelta );
        newPosition = this.mapModelPoint( newPosition );

        // update the position if it is different
        if ( !newPosition.equals( this._positionProperty.value ) ) {
          this._positionProperty.value = newPosition;
        }
      }

      // the optional drag function at the end of any movement
      if ( this._drag ) {
        assert && assert( this._pointer, 'the pointer must be assigned at the start of a drag action' );
        const syntheticEvent = this.createSyntheticEvent( this._pointer! );
        this._drag( syntheticEvent, this );
      }
    }, {
      parameters: [],
      tandem: options.tandem.createTandem( 'dragAction' ),
      phetioDocumentation: 'Emits every time there is some input from a keyboard drag.',
      phetioHighFrequency: true,
      phetioReadOnly: options.phetioReadOnly,
      phetioEventType: EventType.USER
    } );

    this.dragEndAction = new PhetioAction( () => {

      // stop the callback timer
      this.callbackTimer.stop( false );

      const syntheticEvent = this._pointer ? this.createSyntheticEvent( this._pointer ) : null;
      this._end && this._end( syntheticEvent, this );

      this.clearPointer();
    }, {
      parameters: [],
      tandem: options.tandem.createTandem( 'dragEndAction' ),
      phetioDocumentation: 'Emits whenever a keyboard drag ends.',
      phetioReadOnly: options.phetioReadOnly,
      phetioEventType: EventType.USER
    } );

    this._pointerListener = {
      listener: this,
      interrupt: this.interrupt.bind( this )
    };

    this._pointer = null;

    // For dragSpeed implementation, the CallbackTimer will fire every animation frame, so the interval is
    // meant to work at 60 frames per second.
    const interval = this.useDragSpeed ? 1000 / 60 : options.moveOnHoldInterval;
    const delay = this.useDragSpeed ? 0 : options.moveOnHoldDelay;

    this.callbackTimer = new CallbackTimer( {
      delay: delay,
      interval: interval,

      callback: () => {

        let deltaX = 0;
        let deltaY = 0;

        const shiftKeyDown = globalKeyStateTracker.shiftKeyDown;

        let delta: number;
        if ( this.useDragSpeed ) {

          // We know that CallbackTimer is going to fire at the interval so we can use that to get the dt.
          const dt = interval / 1000; // the interval in seconds
          delta = dt * ( shiftKeyDown ? options.shiftDragSpeed : options.dragSpeed );
        }
        else {
          delta = shiftKeyDown ? options.shiftDragDelta : options.dragDelta;
        }

        if ( this.leftKeyDownProperty.value ) {
          deltaX -= delta;
        }
        if ( this.rightKeyDownProperty.value ) {
          deltaX += delta;
        }
        if ( this.upKeyDownProperty.value ) {
          deltaY -= delta;
        }
        if ( this.downKeyDownProperty.value ) {
          deltaY += delta;
        }

        let vectorDelta = new Vector2( deltaX, deltaY );

        // only initiate move if there was some attempted keyboard drag
        if ( !vectorDelta.equals( Vector2.ZERO ) ) {

          // to model coordinates
          if ( options.transform ) {
            const transform = options.transform instanceof Transform3 ? options.transform : options.transform.value;
            vectorDelta = transform.inverseDelta2( vectorDelta );
          }

          this.vectorDelta = vectorDelta;
          this.dragAction.execute();
        }
      }
    } );

    // When any of the movement keys first go down, start the drag operation on the next keydown event (so that
    // the SceneryEvent is available).
    this.isPressedProperty.lazyLink( dragKeysDown => {
      if ( dragKeysDown ) {
        this.startNextKeyboardEvent = true;
      }
      else {

        // In case movement keys are released before we get a keydown event (mostly possible during fuzz testing),
        // don't start the next drag action.
        this.startNextKeyboardEvent = false;
        this.restartTimerNextKeyboardEvent = false;

        this.dragEndAction.execute();
      }
    } );

    // If not the shift key, the drag should start immediately in the direction of the newly pressed key instead
    // of waiting for the next interval. Only important for !useDragSpeed.
    if ( !this.useDragSpeed ) {
      const addStartTimerListener = ( keyProperty: TReadOnlyProperty<boolean> ) => {
        keyProperty.link( keyDown => {
          if ( keyDown ) {
            this.restartTimerNextKeyboardEvent = true;
          }
        } );
      };
      addStartTimerListener( this.leftKeyDownProperty );
      addStartTimerListener( this.rightKeyDownProperty );
      addStartTimerListener( this.upKeyDownProperty );
      addStartTimerListener( this.downKeyDownProperty );
    }

    this._disposeKeyboardDragListener = () => {

      this.leftKeyDownProperty.dispose();
      this.rightKeyDownProperty.dispose();
      this.upKeyDownProperty.dispose();
      this.downKeyDownProperty.dispose();

      this.callbackTimer.dispose();
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
   * Getter for the dragSpeed property, see options.dragSpeed for more info.
   */
  public get dragSpeed(): number { return this._dragSpeed; }

  /**
   * Setter for the dragSpeed property, see options.dragSpeed for more info.
   */
  public set dragSpeed( dragSpeed: number ) { this._dragSpeed = dragSpeed; }

  /**
   * Getter for the shiftDragSpeed property, see options.shiftDragSpeed for more info.
   */
  public get shiftDragSpeed(): number { return this._shiftDragSpeed; }

  /**
   * Setter for the shiftDragSpeed property, see options.shiftDragSpeed for more info.
   */
  public set shiftDragSpeed( shiftDragSpeed: number ) { this._shiftDragSpeed = shiftDragSpeed; }

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
   * Are keys pressed that would move the target Node to the left?
   */
  public movingLeft(): boolean {
    return this.leftKeyDownProperty.value && !this.rightKeyDownProperty.value;
  }

  /**
   * Are keys pressed that would move the target Node to the right?
   */
  public movingRight(): boolean {
    return this.rightKeyDownProperty.value && !this.leftKeyDownProperty.value;
  }

  /**
   * Are keys pressed that would move the target Node up?
   */
  public movingUp(): boolean {
    return this.upKeyDownProperty.value && !this.downKeyDownProperty.value;
  }

  /**
   * Are keys pressed that would move the target Node down?
   */
  public movingDown(): boolean {
    return this.downKeyDownProperty.value && !this.upKeyDownProperty.value;
  }

  /**
   * Get the current target Node of the drag.
   */
  public getCurrentTarget(): Node {
    assert && assert( this.isPressedProperty.value, 'We have no currentTarget if we are not pressed' );
    assert && assert( this._pointer && this._pointer.trail, 'Must have a Pointer with an active trail if we are pressed' );
    return this._pointer!.trail!.lastNode();
  }

  /**
   * Scenery internal. Part of the events API. Do not call directly.
   *
   * Does specific work for the keydown event. This is called during scenery event dispatch, and AFTER any global
   * key state updates. This is important because interruption needs to happen after hotkeyManager has fully processed
   * the key state. And this implementation assumes that the keydown event will happen after Hotkey updates
   * (see startNextKeyboardEvent).
   */
  public override keydown( event: SceneryEvent<KeyboardEvent> ): void {
    super.keydown( event );

    const domEvent = event.domEvent!;

    // If the meta key is down (command key/windows key) prevent movement and do not preventDefault.
    // Meta key + arrow key is a command to go back a page, and we need to allow that. But also, macOS
    // fails to provide keyup events once the meta key is pressed, see
    // http://web.archive.org/web/20160304022453/http://bitspushedaround.com/on-a-few-things-you-may-not-know-about-the-hellish-command-key-and-javascript-events/
    if ( domEvent.metaKey ) {
      return;
    }

    if ( KeyboardUtils.isMovementKey( domEvent ) ) {

      // Prevent a VoiceOver bug where pressing multiple arrow keys at once causes the AT to send the wrong keys
      // through the keyup event - as a workaround, we only allow one arrow key to be down at a time. If two are pressed
      // down, we immediately interrupt.
      if ( platform.safari && this.pressedKeysProperty.value.length > 1 ) {
        this.interrupt();
        return;
      }

      // Finally, in this case we are actually going to drag the object. Prevent default behavior so that Safari
      // doesn't play a 'bonk' sound every arrow key press.
      domEvent.preventDefault();

      // Cannot attach a listener to a Pointer that is already attached.
      if ( this.startNextKeyboardEvent && !event.pointer.isAttached() ) {

        // If there are no movement keys down, attach a listener to the Pointer that will tell the AnimatedPanZoomListener
        // to keep this Node in view
        assert && assert( this._pointer === null, 'Pointer should be null at the start of a drag action' );
        this._pointer = event.pointer as PDOMPointer;
        event.pointer.addInputListener( this._pointerListener, true );

        this.dragStartAction.execute( event );
        this.startNextKeyboardEvent = false;
      }

      if ( this.restartTimerNextKeyboardEvent ) {

        // restart the callback timer
        this.callbackTimer.stop( false );
        this.callbackTimer.start();

        if ( this._moveOnHoldDelay > 0 ) {

          // fire right away if there is a delay - if there is no delay the timer is going to fire in the next
          // animation frame and so it would appear that the object makes two steps in one frame
          this.callbackTimer.fire();
        }

        this.restartTimerNextKeyboardEvent = false;
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
   * If the pointer is set, remove the listener from it and clear the reference.
   */
  private clearPointer(): void {
    if ( this._pointer ) {
      assert && assert( this._pointer.listeners.includes( this._pointerListener ),
        'A reference to the Pointer means it should have the pointerListener' );
      this._pointer.removeInputListener( this._pointerListener );
      this._pointer = null;
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
}

scenery.register( 'KeyboardDragListener', KeyboardDragListener );

export default KeyboardDragListener;