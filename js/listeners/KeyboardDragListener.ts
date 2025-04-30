// Copyright 2019-2025, University of Colorado Boulder

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
 * - Uses stepTimer for smooth, timed updates during drag operations, especially useful in continuous 'dragSpeed'
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

import CallbackTimer from '../../../axon/js/CallbackTimer.js';
import { EnabledComponentOptions } from '../../../axon/js/EnabledComponent.js';
import Property from '../../../axon/js/Property.js';
import stepTimer from '../../../axon/js/stepTimer.js';
import TinyProperty from '../../../axon/js/TinyProperty.js';
import TProperty from '../../../axon/js/TProperty.js';
import TReadOnlyProperty from '../../../axon/js/TReadOnlyProperty.js';
import Bounds2 from '../../../dot/js/Bounds2.js';
import Transform3 from '../../../dot/js/Transform3.js';
import Vector2 from '../../../dot/js/Vector2.js';
import assertMutuallyExclusiveOptions from '../../../phet-core/js/assertMutuallyExclusiveOptions.js';
import optionize, { EmptySelfOptions } from '../../../phet-core/js/optionize.js';
import platform from '../../../phet-core/js/platform.js';
import PickOptional from '../../../phet-core/js/types/PickOptional.js';
import StrictOmit from '../../../phet-core/js/types/StrictOmit.js';
import EventType from '../../../tandem/js/EventType.js';
import PhetioAction from '../../../tandem/js/PhetioAction.js';
import { PhetioObjectOptions } from '../../../tandem/js/PhetioObject.js';
import Tandem from '../../../tandem/js/Tandem.js';
import globalKeyStateTracker from '../accessibility/globalKeyStateTracker.js';
import KeyboardUtils from '../accessibility/KeyboardUtils.js';
import type { OneKeyStroke } from '../input/KeyDescriptor.js';
import PDOMPointer from '../input/PDOMPointer.js';
import SceneryEvent from '../input/SceneryEvent.js';
import type { default as TInputListener, SceneryListenerFunction } from '../input/TInputListener.js';
import Node from '../nodes/Node.js';
import scenery from '../scenery.js';
import type { AllDragListenerOptions } from './AllDragListenerOptions.js';
import type { KeyboardListenerOptions } from './KeyboardListener.js';
import KeyboardListener from './KeyboardListener.js';
import type { SceneryListenerCallback, SceneryListenerNullableCallback } from './PressListener.js';

// 'shift' is not included in any list of keys because we don't want the KeyboardListener to be 'pressed' when only
// the shift key is down. State of the shift key is tracked by the globalKeyStateTracker.

// eslint-disable-next-line @typescript-eslint/no-unused-vars
const keyboardDraggingKeys = [ 'arrowLeft', 'arrowRight', 'arrowUp', 'arrowDown', 'w', 'a', 's', 'd' ] as const;

// eslint-disable-next-line @typescript-eslint/no-unused-vars
const leftRightKeys = [ 'arrowLeft', 'arrowRight', 'a', 'd' ] as const;

// eslint-disable-next-line @typescript-eslint/no-unused-vars
const upDownKeys = [ 'arrowUp', 'arrowDown', 'w', 's' ] as const;

// We still want to start drag operations when the shift modifier key is pressed, even though it is not
// listed in keys for the listener.
const ignoredShiftPattern = 'shift?+';

// KeyDescriptorProperties for each key that can be pressed to move the object.
const A_KEY_STRING_PROPERTY = new Property<OneKeyStroke>( `${ignoredShiftPattern}a` );
const D_KEY_STRING_PROPERTY = new Property<OneKeyStroke>( `${ignoredShiftPattern}d` );
const W_KEY_STRING_PROPERTY = new Property<OneKeyStroke>( `${ignoredShiftPattern}w` );
const S_KEY_STRING_PROPERTY = new Property<OneKeyStroke>( `${ignoredShiftPattern}s` );
const ARROW_LEFT_KEY_STRING_PROPERTY = new Property<OneKeyStroke>( `${ignoredShiftPattern}arrowLeft` );
const ARROW_RIGHT_KEY_STRING_PROPERTY = new Property<OneKeyStroke>( `${ignoredShiftPattern}arrowRight` );
const ARROW_UP_KEY_STRING_PROPERTY = new Property<OneKeyStroke>( `${ignoredShiftPattern}arrowUp` );
const ARROW_DOWN_KEY_STRING_PROPERTY = new Property<OneKeyStroke>( `${ignoredShiftPattern}arrowDown` );

const LEFT_RIGHT_KEY_STRING_PROPERTIES = [ A_KEY_STRING_PROPERTY, D_KEY_STRING_PROPERTY, ARROW_LEFT_KEY_STRING_PROPERTY, ARROW_RIGHT_KEY_STRING_PROPERTY ];
const UP_DOWN_KEY_STRING_PROPERTIES = [ W_KEY_STRING_PROPERTY, S_KEY_STRING_PROPERTY, ARROW_UP_KEY_STRING_PROPERTY, ARROW_DOWN_KEY_STRING_PROPERTY ];
const ALL_KEY_STRING_PROPERTIES = [ ...LEFT_RIGHT_KEY_STRING_PROPERTIES, ...UP_DOWN_KEY_STRING_PROPERTIES ];

type KeyboardDragListenerKeyStroke = typeof keyboardDraggingKeys | typeof leftRightKeys | typeof upDownKeys;

// Possible movement types for this KeyboardDragListener. 2D motion ('both') or 1D motion ('leftRight' or 'upDown').
type KeyboardDragDirection = 'both' | 'leftRight' | 'upDown';
export const KeyboardDragDirectionToKeyStringPropertiesMap = new Map<KeyboardDragDirection, TProperty<OneKeyStroke>[]>( [
  [ 'both', ALL_KEY_STRING_PROPERTIES ],
  [ 'leftRight', LEFT_RIGHT_KEY_STRING_PROPERTIES ],
  [ 'upDown', UP_DOWN_KEY_STRING_PROPERTIES ]
] );

type MapPosition = ( point: Vector2 ) => Vector2;

export type KeyboardDragListenerDOMEvent = KeyboardEvent;
export type KeyboardDragListenerCallback<Listener extends KeyboardDragListener = KeyboardDragListener> = SceneryListenerCallback<Listener, KeyboardDragListenerDOMEvent>;
export type KeyboardDragListenerNullableCallback<Listener extends KeyboardDragListener = KeyboardDragListener> = SceneryListenerNullableCallback<Listener, KeyboardDragListenerDOMEvent>;

type SelfOptions<Listener extends KeyboardDragListener> = {

  // How much the position Property will change in view (parent) coordinates every moveOnHoldInterval. Object will
  // move in discrete steps at this interval. If you would like smoother "animated" motion use dragSpeed
  // instead. dragDelta produces a UX that is more typical for applications but dragSpeed is better for video
  // game-like components. dragDelta and dragSpeed are mutually exclusive options.
  dragDelta?: number;

  // How much the PositionProperty will change in view (parent) coordinates every moveOnHoldInterval while the shift modifier
  // key is pressed. Shift modifier should produce more fine-grained motion so this value needs to be less than
  // dragDelta if provided. Object will move in discrete steps. If you would like smoother "animated" motion use
  // dragSpeed options instead. dragDelta options produce a UX that is more typical for applications but dragSpeed
  // is better for game-like components. dragDelta and dragSpeed are mutually exclusive options.
  shiftDragDelta?: number;

  // While a direction key is held down, the target will move by this amount in view (parent) coordinates every second.
  // This is an alternative way to control motion with keyboard than dragDelta and produces smoother motion for
  // the object. dragSpeed and dragDelta options are mutually exclusive. See dragDelta for more information.
  dragSpeed?: number;

  // While a direction key is held down with the shift modifier key, the target will move by this amount in parent view
  // coordinates every second. Shift modifier should produce more fine-grained motion so this value needs to be less
  // than dragSpeed if provided. This is an alternative way to control motion with keyboard than dragDelta and
  // produces smoother motion for the object. dragSpeed and dragDelta options are mutually exclusive. See dragDelta
  // for more information.
  shiftDragSpeed?: number;

  // Specifies the direction of motion for the KeyboardDragListener. By default, the position Vector2 can change in
  // both directions by pressing the arrow keys. But you can constrain dragging to 1D left-right or up-down motion
  // with this value.
  keyboardDragDirection?: KeyboardDragDirection;

  // Arrow keys must be pressed this long to begin movement set on moveOnHoldInterval, in ms
  moveOnHoldDelay?: number;

  // Time interval at which the object will change position while the arrow key is held down, in ms. This must be larger
  // than 0 to prevent dragging that is based on how often animation-frame steps occur.
  moveOnHoldInterval?: number;

} & AllDragListenerOptions<Listener, KeyboardDragListenerDOMEvent> &

  // Though DragListener is not instrumented, declare these here to support properly passing this to children, see
  // https://github.com/phetsims/tandem/issues/60.
  Pick<PhetioObjectOptions, 'tandem' | 'phetioReadOnly'>;

type ParentOptions = StrictOmit<KeyboardListenerOptions<KeyboardDragListenerKeyStroke>, 'keys'>;

export type KeyboardDragListenerOptions = SelfOptions<KeyboardDragListener> & // Options specific to this class
  PickOptional<ParentOptions, 'focus' | 'blur'> & // Only focus/blur are optional from the superclass
  EnabledComponentOptions; // Other superclass options are allowed

class KeyboardDragListener extends KeyboardListener<KeyboardDragListenerKeyStroke> {

  // See options for documentation
  private readonly _start: KeyboardDragListenerCallback | null;
  private readonly _drag: KeyboardDragListenerCallback | null;
  private readonly _end: KeyboardDragListenerNullableCallback | null;
  private _dragBoundsProperty: TReadOnlyProperty<Bounds2 | null>;
  private readonly _mapPosition: MapPosition | null;
  private readonly _translateNode: boolean;
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
  private dragStartAction: PhetioAction<[ SceneryEvent<KeyboardDragListenerDOMEvent> ]>;
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

  // The vector delta in model coordinates that is used to move the object during a drag operation.
  // If dragSpeed is used, this will be zero in the start callback until there is movement during the
  // animation frame.
  public modelDelta: Vector2 = new Vector2( 0, 0 );

  // The current drag point in the model coordinate frame.
  public modelPoint: Vector2 = new Vector2( 0, 0 );

  // The proposed delta in model coordinates, before mapping or other constraints are applied. If using
  // dragSpeed, this will be zero in the start callback until there is some movement during the animation.
  private vectorDelta: Vector2 = new Vector2( 0, 0 );

  // The callback timer that is used to move the object during a drag operation to support animated motion and
  // motion every moveOnHoldInterval.
  private readonly callbackTimer: CallbackTimer;

  // A listener bound to this that is added to the stepTimer to support dragSpeed implementations. Added and removed
  // as drag starts/ends.
  private readonly boundStepListener: ( ( dt: number ) => void );

  public constructor( providedOptions?: KeyboardDragListenerOptions ) {

    // Use either dragSpeed or dragDelta, cannot use both at the same time.
    assert && assertMutuallyExclusiveOptions( providedOptions, [ 'dragSpeed', 'shiftDragSpeed' ], [ 'dragDelta', 'shiftDragDelta' ] );

    // 'move on hold' timings are only relevant for 'delta' implementations of dragging
    assert && assertMutuallyExclusiveOptions( providedOptions, [ 'dragSpeed' ], [ 'moveOnHoldDelay', 'moveOnHoldInterval' ] );
    assert && assertMutuallyExclusiveOptions( providedOptions, [ 'mapPosition' ], [ 'dragBoundsProperty' ] );

    // If you provide a dragBoundsProperty, you must provide either a positionProperty or use translateNode.
    // KeyboardDragListener operates on deltas and without translating the Node or using a positionProperty there
    // is no way to know where the drag target is or how to constrain it in the drag bounds.
    assert && assert( !providedOptions ||
                      !providedOptions.dragBoundsProperty ||
                      providedOptions.positionProperty || providedOptions.translateNode,
      'If you provide a dragBoundsProperty, you must provide either a positionProperty or use translateNode.' );

    const options = optionize<KeyboardDragListenerOptions, SelfOptions<KeyboardDragListener>, ParentOptions>()( {

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
      translateNode: false,
      start: null,
      drag: null,
      end: null,
      moveOnHoldDelay: 500,
      moveOnHoldInterval: 400,
      tandem: Tandem.REQUIRED,

      // DragListener by default doesn't allow PhET-iO to trigger drag Action events
      phetioReadOnly: true
    }, providedOptions );

    assert && assert( options.shiftDragSpeed <= options.dragSpeed, 'shiftDragSpeed should be less than or equal to dragSpeed, it is intended to provide more fine-grained control' );
    assert && assert( options.shiftDragDelta <= options.dragDelta, 'shiftDragDelta should be less than or equal to dragDelta, it is intended to provide more fine-grained control' );

    const keyStringProperties = KeyboardDragDirectionToKeyStringPropertiesMap.get( options.keyboardDragDirection )!;
    assert && assert( keyStringProperties, 'Invalid keyboardDragDirection' );

    const superOptions = optionize<KeyboardDragListenerOptions, EmptySelfOptions, KeyboardListenerOptions<KeyboardDragListenerKeyStroke>>()( {
      keyStringProperties: keyStringProperties
    }, options );

    super( superOptions );

    // pressedKeysProperty comes from KeyboardListener, and it is used to determine the state of the movement keys.
    // This approach gives more control over the positionProperty in the callbackTimer than using the KeyboardListener
    // callback.
    this.pressedKeyStringPropertiesProperty.link( pressedKeyStringProperties => {
      this.leftKeyDownProperty.value = pressedKeyStringProperties.includes( ARROW_LEFT_KEY_STRING_PROPERTY ) || pressedKeyStringProperties.includes( A_KEY_STRING_PROPERTY );
      this.rightKeyDownProperty.value = pressedKeyStringProperties.includes( ARROW_RIGHT_KEY_STRING_PROPERTY ) || pressedKeyStringProperties.includes( D_KEY_STRING_PROPERTY );
      this.upKeyDownProperty.value = pressedKeyStringProperties.includes( ARROW_UP_KEY_STRING_PROPERTY ) || pressedKeyStringProperties.includes( W_KEY_STRING_PROPERTY );
      this.downKeyDownProperty.value = pressedKeyStringProperties.includes( ARROW_DOWN_KEY_STRING_PROPERTY ) || pressedKeyStringProperties.includes( S_KEY_STRING_PROPERTY );
    } );

    // Mutable attributes declared from options, see options for info, as well as getters and setters.
    this._start = options.start;
    this._drag = options.drag;
    this._end = options.end;
    this._dragBoundsProperty = ( options.dragBoundsProperty || new Property( null ) );
    this._mapPosition = options.mapPosition;
    this._translateNode = options.translateNode;
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

      // If dragging with deltas, we can eagerly compute listener deltas so they are available for
      // the start callback. Otherwise, there will be no motion until the animation frame so deltas
      // are zero.
      if ( !this.useDragSpeed ) {
        const shiftKeyDown = globalKeyStateTracker.shiftKeyDown;
        const delta = shiftKeyDown ? this.shiftDragDelta : this.dragDelta;
        this.computeDeltas( delta );
      }
      else {
        this.computeDeltas( 0 );
      }

      this._start && this._start( event, this );

      // If using dragSpeed, add the listener to the stepTimer to start animated dragging. For dragDelta, the
      // callbackTimer is started every key press, see addStartCallbackTimerListener below.
      if ( this.useDragSpeed ) {
        stepTimer.addListener( this.boundStepListener );
      }
    }, {
      parameters: [ { name: 'event', phetioType: SceneryEvent.SceneryEventIO } ],
      tandem: options.tandem.createTandem( 'dragStartAction' ),
      phetioDocumentation: 'Emits whenever a keyboard drag starts.',
      phetioReadOnly: options.phetioReadOnly,
      phetioEventType: EventType.USER
    } );

    // The drag action only executes when there is actual movement (modelDelta is non-zero). For example, it does
    // NOT execute if conflicting keys are pressed (e.g. left and right arrow keys at the same time). Note that this
    // is expected to be executed from the CallbackTimer. So there will be problems if this can be executed from
    // PhET-iO clients.
    this.dragAction = new PhetioAction( () => {
      assert && assert( this.isPressedProperty.value, 'The listener should not be dragging if not pressed' );

      // Apply translation to the view coordinate frame.
      if ( this._translateNode ) {
        let newPosition = this.getCurrentTarget().translation.plus( this.vectorDelta );
        newPosition = this.mapModelPoint( newPosition );
        this.getCurrentTarget().translation = newPosition;

        this.modelPoint = this.parentToModelPoint( newPosition );
      }

      // Synchronize with model position.
      if ( this._positionProperty ) {
        let newPosition = this._positionProperty.value.plus( this.modelDelta );
        newPosition = this.mapModelPoint( newPosition );

        this.modelPoint = newPosition;

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
      if ( this.useDragSpeed ) {
        stepTimer.removeListener( this.boundStepListener );
      }
      else {
        this.callbackTimer.stop( false );
      }

      const syntheticEvent = this._pointer ? this.createSyntheticEvent( this._pointer ) : null;
      this._end && this._end( syntheticEvent, this );

      this.clearPointer();

      // The listener deltas go back to zero at the end of interaction.
      this.computeDeltas( 0 );
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

    // CallbackTimer will be used to support dragDelta callback intervals. It will be restarted whenever there is a
    // new key press so that the object moves every time there is user input. It is stopped when all keys are released.
    this.callbackTimer = new CallbackTimer( {
      callback: () => {
        const shiftKeyDown = globalKeyStateTracker.shiftKeyDown;
        const delta = shiftKeyDown ? options.shiftDragDelta : options.dragDelta;
        this.moveFromDelta( delta );
      },

      delay: options.moveOnHoldDelay,
      interval: options.moveOnHoldInterval
    } );

    // A listener is added to the stepTimer to support dragSpeed. Does not use CallbackTimer because CallbackTimer
    // uses setInterval and may not be called every frame which results in choppy motion, see
    // https://github.com/phetsims/scenery/issues/1638. It is added to the stepTimer when the drag starts and removed
    // when the drag ends.
    this.boundStepListener = this.stepForSpeed.bind( this );

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

        if ( this.isDragging() ) {
          this.dragEndAction.execute();
        }
      }
    } );

    // If not the shift key, the drag should start immediately in the direction of the newly pressed key instead
    // of waiting for the next interval. Only important for !useDragSpeed (using CallbackTimer).
    if ( !this.useDragSpeed ) {
      const addStartCallbackTimerListener = ( keyProperty: TReadOnlyProperty<boolean> ) => {
        keyProperty.link( keyDown => {
          if ( keyDown ) {
            this.restartTimerNextKeyboardEvent = true;
          }
        } );
      };
      addStartCallbackTimerListener( this.leftKeyDownProperty );
      addStartCallbackTimerListener( this.rightKeyDownProperty );
      addStartCallbackTimerListener( this.upKeyDownProperty );
      addStartCallbackTimerListener( this.downKeyDownProperty );
    }

    this._disposeKeyboardDragListener = () => {

      this.leftKeyDownProperty.dispose();
      this.rightKeyDownProperty.dispose();
      this.upKeyDownProperty.dispose();
      this.downKeyDownProperty.dispose();

      this.callbackTimer.dispose();
      if ( stepTimer.hasListener( this.boundStepListener ) ) {
        stepTimer.removeListener( this.boundStepListener );
      }
    };
  }

  /**
   * Calculates a delta for movement from the time step. Only used for `dragSpeed`. This is bound and added to
   * the stepTimer when dragging starts.
   */
  private stepForSpeed( dt: number ): void {
    assert && assert( this.useDragSpeed, 'This method should only be called when using dragSpeed' );

    const shiftKeyDown = globalKeyStateTracker.shiftKeyDown;
    const delta = dt * ( shiftKeyDown ? this.shiftDragSpeed : this.dragSpeed );
    this.moveFromDelta( delta );
  }

  /**
   * Given a delta from dragSpeed or dragDelta, determine the direction of movement and move the object accordingly
   * by using the dragAction.
   */
  private moveFromDelta( delta: number ): void {
    this.computeDeltas( delta );

    // only initiate move if there was some attempted keyboard drag
    if ( !this.vectorDelta.equals( Vector2.ZERO ) ) {
      this.dragAction.execute();
    }
  }

  /**
   * Compute vectorDelta and modelDelta for this listener from keys pressed and a provided delta.
   */
  private computeDeltas( delta: number ): void {
    let deltaX = 0;
    let deltaY = 0;

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
    this.vectorDelta = new Vector2( deltaX, deltaY );

    // Convert the proposed delta to model coordinates
    if ( this.transform ) {
      const transform = this.transform instanceof Transform3 ? this.transform : this.transform.value;
      this.modelDelta = transform.inverseDelta2( this.vectorDelta );
    }
    else {
      this.modelDelta = this.vectorDelta;
    }
  }

  /**
   * Convert a point in the view (parent) coordinate frame to the model coordinate frame.
   */
  private parentToModelPoint( parentPoint: Vector2 ): Vector2 {
    if ( this.transform ) {
      const transform = this.transform instanceof Transform3 ? this.transform : this.transform.value;
      return transform.inverseDelta2( parentPoint );
    }
    return parentPoint;
  }

  private localToParentPoint( localPoint: Vector2 ): Vector2 {
    const target = this.getCurrentTarget();
    return target.localToParentPoint( localPoint );
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
   * Returns true when this listener is currently dragging. The pointer must be assigned (drag started) and it
   * must be attached to this _pointerListener (otherwise it is interacting with another target).
   */
  private isDragging(): boolean {
    return !!this._pointer && this._pointer.attachedListener === this._pointerListener;
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
      if ( platform.safari && this.pressedKeyStringPropertiesProperty.value.length > 1 ) {
        this.interrupt();
        return;
      }

      // Finally, in this case we are actually going to drag the object. Prevent default behavior so that Safari
      // doesn't play a 'bonk' sound every arrow key press.
      domEvent.preventDefault();

      // Cannot attach a listener to a Pointer that is already attached. This needs to happen before
      // firing the callback timer, which can initiate a call to drag().
      if ( this.startNextKeyboardEvent && !event.pointer.isAttached() ) {

        // If there are no movement keys down, attach a listener to the Pointer that will tell the AnimatedPanZoomListener
        // to keep this Node in view
        assert && assert( this._pointer === null, 'Pointer should be null at the start of a drag action' );
        this._pointer = event.pointer as PDOMPointer;
        event.pointer.addInputListener( this._pointerListener, true );

        this.dragStartAction.execute( event );
        this.startNextKeyboardEvent = false;
      }

      // If the drag is already started, restart the callback timer on the next keydown event. The Pointer must
      // be attached to this._pointerListener (already dragging) and not another listener (keyboard is interacting
      // with another target).
      if ( this.restartTimerNextKeyboardEvent && this.isDragging() ) {

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

  // (scenery-internal) Tagged for instanceof-like checks that do not require an import
  public readonly _isKeyboardDragListener = true;

  /**
   * Make eligible for garbage collection.
   */
  public override dispose(): void {
    this.interrupt();
    this._disposeKeyboardDragListener();
    super.dispose();
  }

  /**
   * Creates an input listener that forwards interaction to another Node. Transfers focus to the target Node.
   * Focus is set after the callback so any setup can be done first (like creating a new target Node).
   *
   * Most useful for forwarding input from an icon to another draggable Node.
   *
   * @param targetNode - The Node to forward focus to.
   * @param click - Function that will do any other setup work.
   */
  public static createForwardingListener( targetNode: Node, click: SceneryListenerFunction<MouseEvent | KeyboardEvent> ): TInputListener {
    return {
      click( event ) {
        if ( event.canStartPress() ) {
          click( event );

          assert && assert( targetNode.focusable, 'You are trying to forward keyboard dragging to a Node that is not focsauble.' );
          targetNode.focus();
        }
      },
      keydown( event ) {

        // Handle enter/space so that the forwarding Node is not required to use `tagName: button` to receive
        // click events.
        if ( KeyboardUtils.isAnyKeyEvent( event.domEvent, [ KeyboardUtils.KEY_ENTER, KeyboardUtils.KEY_SPACE ] ) ) {

          // Click actually uses a MouseEvent, but that is not relevant here.
          this.click!( event as unknown as SceneryEvent<MouseEvent> );
        }
      }
    };
  }
}

scenery.register( 'KeyboardDragListener', KeyboardDragListener );

export default KeyboardDragListener;