// Copyright 2017-2023, University of Colorado Boulder

/**
 * A listener for common button usage, providing the fire() method/callback and helpful properties. NOTE that it doesn't
 * need to be an actual button (or look like a button), this is useful whenever that type of "fire" behavior is helpful.
 *
 * For example usage, see scenery/examples/input.html. Usually you can just pass a fire callback and things work.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import CallbackTimer from '../../../axon/js/CallbackTimer.js';
import Emitter from '../../../axon/js/Emitter.js';
import TEmitter from '../../../axon/js/TEmitter.js';
import optionize from '../../../phet-core/js/optionize.js';
import EventType from '../../../tandem/js/EventType.js';
import PhetioObject from '../../../tandem/js/PhetioObject.js';
import Tandem from '../../../tandem/js/Tandem.js';
import NullableIO from '../../../tandem/js/types/NullableIO.js';
import { TInputListener, Node, PressListener, PressListenerOptions, scenery, SceneryEvent } from '../imports.js';

type SelfOptions = {
  // Called as fire() when the button is fired.
  fire?: ( event: SceneryEvent<MouseEvent | TouchEvent | PointerEvent | FocusEvent | KeyboardEvent> ) => void;

  // If true, the button will fire when the button is pressed. If false, the button will fire when the
  // button is released while the pointer is over the button.
  fireOnDown?: boolean;

  // fire-on-hold feature, see https://github.com/phetsims/scenery/issues/1004
  // TODO: these options are not supported with PDOM interaction, see https://github.com/phetsims/scenery/issues/1117
  fireOnHold?: boolean; // is the fire-on-hold feature enabled?
  fireOnHoldDelay?: number; // start to fire continuously after pressing for this long (milliseconds)
  fireOnHoldInterval?: number; // fire continuously at this interval (milliseconds)
};

export type FireListenerOptions<Listener extends FireListener> = SelfOptions & PressListenerOptions<Listener>;

export default class FireListener extends PressListener implements TInputListener {

  private _fireOnDown: boolean;
  private firedEmitter: TEmitter<[ SceneryEvent<MouseEvent | TouchEvent | PointerEvent | FocusEvent | KeyboardEvent> | null ]>;
  private _timer?: CallbackTimer;

  public constructor( providedOptions?: FireListenerOptions<FireListener> ) {
    const options = optionize<FireListenerOptions<FireListener>, SelfOptions, PressListenerOptions<FireListener>>()( {
      fire: _.noop,
      fireOnDown: false,
      fireOnHold: false,
      fireOnHoldDelay: 400,
      fireOnHoldInterval: 100,

      // phet-io
      tandem: Tandem.REQUIRED,

      // Though FireListener is not instrumented, declare these here to support properly passing this to children
      phetioReadOnly: PhetioObject.DEFAULT_OPTIONS.phetioReadOnly
    }, providedOptions );

    assert && assert( typeof options.fire === 'function', 'The fire callback should be a function' );
    assert && assert( typeof options.fireOnDown === 'boolean', 'fireOnDown should be a boolean' );

    // @ts-expect-error TODO see https://github.com/phetsims/phet-core/issues/128
    super( options );

    this._fireOnDown = options.fireOnDown;

    this.firedEmitter = new Emitter<[ SceneryEvent<MouseEvent | TouchEvent | PointerEvent | FocusEvent | KeyboardEvent> | null ]>( {
      tandem: options.tandem.createTandem( 'firedEmitter' ),
      phetioEventType: EventType.USER,
      phetioReadOnly: options.phetioReadOnly,
      phetioDocumentation: 'Emits at the time that the listener fires',
      parameters: [ {
        name: 'event',
        phetioType: NullableIO( SceneryEvent.SceneryEventIO )
      } ]
    } );
    // @ts-expect-error TODO Emitter
    this.firedEmitter.addListener( options.fire );

    // Create a timer to handle the optional fire-on-hold feature.
    // When that feature is enabled, calling this.fire is delegated to the timer.
    if ( options.fireOnHold ) {
      this._timer = new CallbackTimer( {
        callback: this.fire.bind( this, null ), // Pass null for fire-on-hold events
        delay: options.fireOnHoldDelay,
        interval: options.fireOnHoldInterval
      } );
    }
  }

  /**
   * Fires any associated button fire callback.
   *
   * NOTE: This is safe to call on the listener externally.
   */
  public fire( event: SceneryEvent<MouseEvent | TouchEvent | PointerEvent | FocusEvent | KeyboardEvent> | null ): void {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'FireListener fire' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    this.firedEmitter.emit( event );

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  /**
   * Presses the button.
   *
   * NOTE: This is safe to call externally in order to attempt to start a press. fireListener.canPress( event ) can
   * be used to determine whether this will actually start a press.
   *
   * @param event
   * @param [targetNode] - If provided, will take the place of the targetNode for this call. Useful for
   *                              forwarded presses.
   * @param [callback] - to be run at the end of the function, but only on success
   * @returns success - Returns whether the press was actually started
   */
  public override press( event: SceneryEvent<MouseEvent | TouchEvent | PointerEvent | FocusEvent | KeyboardEvent>, targetNode?: Node, callback?: () => void ): boolean {
    return super.press( event, targetNode, () => {
      // This function is only called on success
      if ( this._fireOnDown ) {
        this.fire( event );
      }
      if ( this._timer ) {
        this._timer.start();
      }
      callback && callback();
    } );
  }

  /**
   * Releases the button.
   *
   * NOTE: This can be safely called externally in order to force a release of this button (no actual 'up' event is
   * needed). If the cancel/interrupt behavior is more preferable (will not fire the button), then call interrupt()
   * on this listener instead.
   *
   * @param [event] - scenery event if there was one
   * @param [callback] - called at the end of the release
   */
  public override release( event?: SceneryEvent<MouseEvent | TouchEvent | PointerEvent | FocusEvent | KeyboardEvent>, callback?: () => void ): void {
    super.release( event, () => {
      // Notify after the rest of release is called in order to prevent it from triggering interrupt().
      const shouldFire = !this._fireOnDown && this.isHoveringProperty.value && !this.interrupted;
      if ( this._timer ) {
        this._timer.stop( shouldFire );
      }
      else if ( shouldFire ) {
        this.fire( event || null );
      }
      callback && callback();
    } );
  }

  /**
   * Clicks the listener, pressing it and releasing it immediately. Part of the scenery input API, triggered from PDOM
   * events for accessibility.
   *
   * Click does not involve the FireListener timer because it is a discrete event that
   * presses and releases the listener immediately. This behavior is a limitation imposed
   * by screen reader technology. See PressListener.click for more information.
   *
   * NOTE: This can be safely called externally in order to click the listener, event is not required.
   * fireListener.canClick() can be used to determine if this will actually trigger a click.
   *
   * @param [event]
   * @param [callback] - called at the end of the click
   */
  public override click( event: SceneryEvent<MouseEvent> | null, callback?: () => void ): boolean {
    return super.click( event, () => {

      // don't click if listener was interrupted before this callback
      if ( !this.interrupted ) {
        this.fire( event );
      }
      callback && callback();
    } );
  }

  /**
   * Interrupts the listener, releasing it (canceling behavior).
   *
   * This effectively releases/ends the press, and sets the `interrupted` flag to true while firing these events
   * so that code can determine whether a release/end happened naturally, or was canceled in some way.
   *
   * This can be called manually, but can also be called through node.interruptSubtreeInput().
   */
  public override interrupt(): void {
    super.interrupt();

    this._timer && this._timer.stop( false ); // Stop the timer, don't fire if we haven't already
  }

  public override dispose(): void {
    this.firedEmitter.dispose();
    this._timer && this._timer.dispose();

    super.dispose();
  }
}

scenery.register( 'FireListener', FireListener );
