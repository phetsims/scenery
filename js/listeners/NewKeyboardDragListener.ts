// Copyright 2019-2024, University of Colorado Boulder
//
// @author Jesse Greenberg

import { EnglishStringToCodeMap, globalKeyStateTracker, KeyboardListener, OneKeyStroke, scenery } from '../imports.js';
import Vector2 from '../../../dot/js/Vector2.js';
import assertMutuallyExclusiveOptions from '../../../phet-core/js/assertMutuallyExclusiveOptions.js';
import CallbackTimer from '../../../axon/js/CallbackTimer.js';
import TinyProperty from '../../../axon/js/TinyProperty.js';
import DerivedProperty from '../../../axon/js/DerivedProperty.js';
import Property from '../../../axon/js/Property.js';
import Transform3 from '../../../dot/js/Transform3.js';

export default class NewKeyboardDragListener extends KeyboardListener<OneKeyStroke[]> {
  private leftKeyDownProperty: TinyProperty<boolean>;
  private rightKeyDownProperty: TinyProperty<boolean>;
  private upKeyDownProperty: TinyProperty<boolean>;
  private downKeyDownProperty: TinyProperty<boolean>;
  private shiftKeyDownProperty: TinyProperty<boolean>;

  private callbackTimer: CallbackTimer;

  private useDragSpeed: boolean;

  private positionProperty: Property | null;
  private dragDelta: number;
  private shiftDragDelta: number;
  private moveOnHoldDelay: number;

  public constructor( providedOptions ) {

    assert && assertMutuallyExclusiveOptions( providedOptions, [ 'dragSpeed', 'shiftDragSpeed' ], [ 'dragDelta', 'shiftDragDelta' ] );

    const options = _.merge( {
      positionProperty: null,
      dragDelta: 10,
      shiftDragDelta: 5,
      moveOnHoldDelay: 500,
      moveOnHoldInterval: 400,

      keyboardDragDirection: 'both',

      transform: null,

      dragSpeed: 0,
      shiftDragSpeed: 0

    }, providedOptions );

    let keys: OneKeyStroke[];
    if ( options.keyboardDragDirection === 'both' ) {
      keys = [ 'arrowLeft', 'arrowRight', 'arrowUp', 'arrowDown', 'w', 'a', 's', 'd', 'shift' ];
    }
    else if ( options.keyboardDragDirection === 'leftRight' ) {
      keys = [ 'arrowLeft', 'arrowRight', 'a', 'd', 'shift' ];
    }
    else if ( options.keyboardDragDirection === 'upDown' ) {
      keys = [ 'arrowUp', 'arrowDown', 'w', 's', 'shift' ];
    }
    else {
      throw new Error( 'unhandled keyboardDragDirection' );
    }

    // We need our own interval for smooth dragging across multiple keys.
    // Use KeyboardListener for adding event listeners.
    // Use stepTimer for updating the PositionProperty.
    // use globalKeyStateTracker to watch the keystate.

    super(
      {
        keys: keys,
        listenerFireTrigger: 'both',
        allowExtraModifierKeys: true,
        callback: ( event, keysPressed, listener ) => {
          if ( listener.keysDown ) {
            if ( keysPressed === 'shift' ) {
              this.shiftKeyDownProperty.value = true;
            }
            if ( keysPressed === ( 'arrowLeft' ) || keysPressed === ( 'a' ) ) {
              this.leftKeyDownProperty.value = true;
            }
            if ( keysPressed === ( 'arrowRight' ) || keysPressed === ( 'd' ) ) {
              this.rightKeyDownProperty.value = true;
            }
            if ( keysPressed === ( 'arrowUp' ) || keysPressed === ( 'w' ) ) {
              this.upKeyDownProperty.value = true;
            }
            if ( keysPressed === ( 'arrowDown' ) || keysPressed === ( 's' ) ) {
              this.downKeyDownProperty.value = true;
            }
          }
          else {
            if ( keysPressed === ( 'arrowLeft' ) || keysPressed === ( 'a' ) ) {
              this.leftKeyDownProperty.value = false;
            }
            if ( keysPressed === ( 'arrowRight' ) || keysPressed === ( 'd' ) ) {
              this.rightKeyDownProperty.value = false;
            }
            if ( keysPressed === ( 'arrowUp' ) || keysPressed === ( 'w' ) ) {
              this.upKeyDownProperty.value = false;
            }
            if ( keysPressed === ( 'arrowDown' ) || keysPressed === ( 's' ) ) {
              this.downKeyDownProperty.value = false;
            }
            if ( keysPressed === ( 'shift' ) ) {
              this.shiftKeyDownProperty.value = false;
            }
          }
        }
      }
    );

    // Since dragSpeed and dragDelta are mutually-exclusive drag implementations, a value for either one of these
    // options indicates we should use a speed implementation for dragging.
    this.useDragSpeed = options.dragSpeed > 0 || options.shiftDragSpeed > 0;

    this.leftKeyDownProperty = new TinyProperty( false );
    this.rightKeyDownProperty = new TinyProperty( false );
    this.upKeyDownProperty = new TinyProperty( false );
    this.downKeyDownProperty = new TinyProperty( false );
    this.shiftKeyDownProperty = new TinyProperty( false );

    this.positionProperty = options.positionProperty;
    this.dragDelta = options.dragDelta;
    this.shiftDragDelta = options.shiftDragDelta;
    this.moveOnHoldDelay = options.moveOnHoldDelay;

    const dragKeysDownProperty = new DerivedProperty( [ this.leftKeyDownProperty, this.rightKeyDownProperty, this.upKeyDownProperty, this.downKeyDownProperty ], ( left, right, up, down ) => {
      return left || right || up || down;
    } );

    const interval = this.useDragSpeed ? 1000 / 60 : options.moveOnHoldInterval;
    const delay = this.useDragSpeed ? 0 : options.moveOnHoldDelay;

    this.callbackTimer = new CallbackTimer( {
      delay: delay,
      interval: interval,

      callback: () => {

        let deltaX = 0;
        let deltaY = 0;

        let delta = 0;
        if ( this.useDragSpeed ) {

          // TODO: Is there a better way to get this dt? Its nice that setInterval accounts for 'leftover' time, see #444
          // so that errors dont accumulate. But it would be nice to have a way to get the actual dt.
          const dt = interval / 1000; // the interval in seconds
          delta = dt * ( this.shiftKeyDownProperty.value ? options.shiftDragSpeed : options.dragSpeed );
        }
        else {
          delta = this.shiftKeyDownProperty.value ? options.shiftDragDelta : options.dragDelta;
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

        if ( options.positionProperty ) {
          let vectorDelta = new Vector2( deltaX, deltaY );

          // to model coordinates
          if ( options.transform ) {
            const transform = options.transform instanceof Transform3 ? options.transform : options.transform.value;
            vectorDelta = transform.inverseDelta2( vectorDelta );
          }

          options.positionProperty.set( options.positionProperty.get().plus( vectorDelta ) );
        }
      }
    } );

    // When the drag keys are down, start the callback timer. When they are up, stop the callback timer.
    dragKeysDownProperty.link( dragKeysDown => {
      if ( dragKeysDown ) {

        if ( this.useDragSpeed ) {
          this.callbackTimer.start();
        }

        // this is where we call the optional start callback
      }
      else {

        // when keys are no longer pressed, stop the timer
        this.callbackTimer.stop( false );

        // this is where we call the optional end callback
      }
    } );


    // If using discrete steps, the CallbackTimer is restarted every key press
    if ( !this.useDragSpeed ) {

      // If not the shift key, we need to move immediately in that direction. Only important for !useDragSpeed.
      // This is done oustide of the CallbackTimer listener because we only want to move immediately
      // in the direction of the pressed key.
      const addStartTimerListener = keyProperty => {
        keyProperty.link( keyDown => {
          if ( keyDown ) {

            // restart the callback timer
            this.callbackTimer.stop( false );
            this.callbackTimer.start();

            if ( this.moveOnHoldDelay > 0 ) {

              // fire right away if there is a delay - if there is no delay the timer is going to fire in the next
              // animation frame and so it would appear that the object makes two steps in one frame
              this.callbackTimer.fire();
            }
          }
        } );
      };
      addStartTimerListener( this.leftKeyDownProperty );
      addStartTimerListener( this.rightKeyDownProperty );
      addStartTimerListener( this.upKeyDownProperty );
      addStartTimerListener( this.downKeyDownProperty );
    }
  }

  public override interrupt(): void {
    super.interrupt();

    // Setting these to false doesn't work with the interrupt strategy. They are set to false and the super
    // is interrupted. Then we will get a new keydown event in the super, which will call subclass calbacks,
    // and set these to true again in a later event.
    this.leftKeyDownProperty.value = false;
    this.rightKeyDownProperty.value = false;
    this.upKeyDownProperty.value = false;
    this.downKeyDownProperty.value = false;
    this.shiftKeyDownProperty.value = false;

    this.callbackTimer.stop( false );

  }
}

scenery.register( 'NewKeyboardDragListener', NewKeyboardDragListener );