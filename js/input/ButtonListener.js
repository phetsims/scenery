// Copyright 2013-2022, University of Colorado Boulder

/**
 * Basic button handling.
 *
 * Uses 4 states:
 * up: mouse not over, not pressed
 * over: mouse over, not pressed
 * down: mouse over, pressed
 * out: mouse not over, pressed
 *
 * TODO: offscreen handling
 * TODO: fix enter/exit edge cases for moving nodes or add/remove child, and when touches are created
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import deprecationWarning from '../../../phet-core/js/deprecationWarning.js';
import merge from '../../../phet-core/js/merge.js';
import EventType from '../../../tandem/js/EventType.js';
import Tandem from '../../../tandem/js/Tandem.js';
import IOType from '../../../tandem/js/types/IOType.js';
import { DownUpListener, scenery } from '../imports.js';

/**
 * @deprecated - please use FireListener for new code (set up for the `fire` callback to be easy, and has Properties
 * that can be checked for the other states or complicated cases)
 */
class ButtonListener extends DownUpListener {
  /**
   * Options for the ButtonListener:
   *
   * mouseButton: 0
   * fireOnDown: false // default is to fire on 'up' after 'down', but passing fireOnDown: true will fire on 'down' instead
   * up: null          // Called on an 'up' state change, as up( event, oldState )
   * over: null        // Called on an 'over' state change, as over( event, oldState )
   * down: null        // Called on an 'down' state change, as down( event, oldState )
   * out: null         // Called on an 'out' state change, as out( event, oldState )
   * fire: null        // Called on a state change to/from 'down' (depending on fireOnDown), as fire( event ). Called after the triggering up/over/down event.
   */
  constructor( options ) {
    assert && deprecationWarning( 'ButtonListener is deprecated, please use FireListener instead' );


    options = merge( {

      // When running in PhET-iO brand, the tandem must be supplied
      tandem: Tandem.OPTIONAL,
      phetioType: ButtonListener.ButtonListenerIO,
      phetioState: false,
      phetioEventType: EventType.USER
    }, options );

    // TODO: pass through options
    super( {
      tandem: options.tandem,
      phetioType: options.phetioType,
      phetioState: options.phetioState,

      mouseButton: options.mouseButton || 0, // forward the mouse button, default to 0 (LMB)

      // parameter to DownUpListener, NOT an input listener itself
      down: ( event, trail ) => {
        this.setButtonState( event, 'down' );
      },

      // parameter to DownUpListener, NOT an input listener itself
      up: ( event, trail ) => {
        this.setButtonState( event, this._overCount > 0 ? 'over' : 'up' );
      }
    } );

    // @public {string} - 'up', 'over', 'down' or 'out'
    this.buttonState = 'up';

    // @private {number} - how many pointers are over us (track a count, so we can handle multiple pointers gracefully)
    this._overCount = 0;

    // @private {Object} - store the options object so we can call the callbacks
    this._buttonOptions = options;
  }

  /**
   * @public
   *
   * @param {SceneryEvent} event
   * @param {string} state
   */
  setButtonState( event, state ) {
    if ( state !== this.buttonState ) {
      sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent(
        `ButtonListener state change to ${state} from ${this.buttonState} for ${this.downTrail ? this.downTrail.toString() : this.downTrail}` );
      const oldState = this.buttonState;

      this.buttonState = state;

      if ( this._buttonOptions[ state ] ) {

        // Record this event to the phet-io data stream, including all downstream events as nested children
        this.phetioStartEvent( state );

        // Then invoke the callback
        this._buttonOptions[ state ]( event, oldState );

        this.phetioEndEvent();
      }

      if ( this._buttonOptions.fire &&
           this._overCount > 0 &&
           !this.interrupted &&
           ( this._buttonOptions.fireOnDown ? ( state === 'down' ) : ( oldState === 'down' ) ) ) {

        // Record this event to the phet-io data stream, including all downstream events as nested children
        this.phetioStartEvent( 'fire' );

        // Then fire the event
        this._buttonOptions.fire( event );

        this.phetioEndEvent();
      }
    }
  }

  /**
   * @public (scenery-internal)
   *
   * @param {SceneryEvent} event
   */
  enter( event ) {
    sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent(
      `ButtonListener enter for ${this.downTrail ? this.downTrail.toString() : this.downTrail}` );
    this._overCount++;
    if ( this._overCount === 1 ) {
      this.setButtonState( event, this.isDown ? 'down' : 'over' );
    }
  }

  /**
   * @public (scenery-internal)
   *
   * @param {SceneryEvent} event
   */
  exit( event ) {
    sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent(
      `ButtonListener exit for ${this.downTrail ? this.downTrail.toString() : this.downTrail}` );
    assert && assert( this._overCount > 0, 'Exit events not matched by an enter' );
    this._overCount--;
    if ( this._overCount === 0 ) {
      this.setButtonState( event, this.isDown ? 'out' : 'up' );
    }
  }

  /**
   * Called from "focus" events (part of the Scenery listener API). On focus the PDOMPointer is over the node
   * with the attached listener, so add to the over count.
   * @private
   *
   * @param {SceneryEvent} event
   */
  focus( event ) {
    this.enter( event );
  }

  /**
   * Called from "blur" events (part of the Scenery listener API). On blur, the PDOMPointer leaves the node
   * with this listener so reduce the over count.
   * @private
   *
   * @param {SceneryEvent} event
   */
  blur( event ) {
    this.exit( event );
  }

  /**
   * Called with "click" events (part of the Scenery listener API). Typically will be called from a keyboard
   * or assistive device.
   *
   * There are no `keyup` or `keydown` events when an assistive device is active. So we respond generally
   * to the single `click` event, which indicates a logical activation of this button.
   * TODO: This may change after https://github.com/phetsims/scenery/issues/1117 is done, at which point
   * `click` should likely be replaced by `keydown` and `keyup` listeners.
   * @private
   *
   * @param {SceneryEvent} event
   */
  click( event ) {
    this.setButtonState( event, 'down' );
    this.setButtonState( event, 'up' );
  }
}

scenery.register( 'ButtonListener', ButtonListener );

ButtonListener.ButtonListenerIO = new IOType( 'ButtonListenerIO', {
  valueType: ButtonListener,
  documentation: 'Button listener',
  events: [ 'up', 'over', 'down', 'out', 'fire' ]
} );

export default ButtonListener;