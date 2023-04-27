// Copyright 2013-2023, University of Colorado Boulder

/**
 * Basic down/up pointer handling for a Node, so that it's easy to handle buttons
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import deprecationWarning from '../../../phet-core/js/deprecationWarning.js';
import merge from '../../../phet-core/js/merge.js';
import PhetioObject from '../../../tandem/js/PhetioObject.js';
import { EventContext, Mouse, scenery, SceneryEvent, Trail } from '../imports.js';

/**
 * @deprecated - use PressListener instead
 */
class DownUpListener extends PhetioObject {
  /**
   * The 'trail' parameter passed to down/upInside/upOutside will end with the node to which this DownUpListener has
   * been added.
   *
   * Allowed options: {
   *    mouseButton: 0  // The mouse button to use: left: 0, middle: 1, right: 2, see
   *                    // https://developer.mozilla.org/en-US/docs/Web/API/MouseEvent
   *    down: null      // down( event, trail ) is called when the pointer is pressed down on this node
   *                    // (and another pointer is not already down on it).
   *    up: null        // up( event, trail ) is called after 'down', regardless of the pointer's current location.
   *                    // Additionally, it is called AFTER upInside or upOutside, whichever is relevant
   *    upInside: null  // upInside( event, trail ) is called after 'down', when the pointer is released inside
   *                    // this node (it or a descendant is the top pickable node under the pointer)
   *    upOutside: null // upOutside( event, trail ) is called after 'down', when the pointer is released outside
   *                    // this node (it or a descendant is the not top pickable node under the pointer, even if the
   *                    // same instance is still directly under the pointer)
   * }
   *
   * @param {Object} [options]
   */
  constructor( options ) {
    assert && deprecationWarning( 'DownUpListener is deprecated, please use PressListener instead' );


    options = merge( {
      mouseButton: 0 // allow a different mouse button
    }, options );

    super( options );

    // @private {Object}
    this.options = options;

    // @public {boolean} - whether this listener is down
    this.isDown = false;

    // @public {Node|null} - 'up' is handled via a pointer lister, which will have null currentTarget, so save the
    // 'down' currentTarget
    this.downCurrentTarget = null;

    // @public {Trail|null}
    this.downTrail = null;

    // @public {Pointer|null}
    this.pointer = null;

    // @public {boolean}
    this.interrupted = false;

    // @private {function} - this listener gets added to the pointer on a 'down'
    this.downListener = {
      // mouse/touch up
      up: event => {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( `DownUpListener (pointer) up for ${this.downTrail.toString()}` );
        sceneryLog && sceneryLog.InputListener && sceneryLog.push();

        assert && assert( event.pointer === this.pointer );
        if ( !( event.pointer instanceof Mouse ) || event.domEvent.button === this.options.mouseButton ) {
          this.buttonUp( event );
        }

        sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
      },

      // interruption of this Pointer listener
      interrupt: () => {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( `DownUpListener (pointer) interrupt for ${this.downTrail.toString()}` );
        sceneryLog && sceneryLog.InputListener && sceneryLog.push();

        this.interrupt();

        sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
      },

      // touch cancel
      cancel: event => {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( `DownUpListener (pointer) cancel for ${this.downTrail.toString()}` );
        sceneryLog && sceneryLog.InputListener && sceneryLog.push();

        assert && assert( event.pointer === this.pointer );
        this.buttonUp( event );

        sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
      }
    };
  }

  /**
   * @private
   *
   * @param {SceneryEvent} event
   */
  buttonDown( event ) {
    // already down from another pointer, don't do anything
    if ( this.isDown ) { return; }

    // ignore other mouse buttons
    if ( event.pointer instanceof Mouse && event.domEvent.button !== this.options.mouseButton ) { return; }

    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DownUpListener buttonDown' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    // add our listener so we catch the up wherever we are
    event.pointer.addInputListener( this.downListener );

    this.isDown = true;
    this.downCurrentTarget = event.currentTarget;
    this.downTrail = event.trail.subtrailTo( event.currentTarget, false );
    this.pointer = event.pointer;

    if ( this.options.down ) {
      this.options.down( event, this.downTrail );
    }

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  /**
   * @private
   *
   * @param {SceneryEvent} event
   */
  buttonUp( event ) {
    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DownUpListener buttonUp' );
    sceneryLog && sceneryLog.InputListener && sceneryLog.push();

    this.isDown = false;
    this.pointer.removeInputListener( this.downListener );

    const currentTargetSave = event.currentTarget;
    event.currentTarget = this.downCurrentTarget; // up is handled by a pointer listener, so currentTarget would be null.
    if ( this.options.upInside || this.options.upOutside ) {
      const trailUnderPointer = event.trail;

      // TODO: consider changing this so that it just does a hit check and ignores anything in front?
      const isInside = trailUnderPointer.isExtensionOf( this.downTrail, true ) && !this.interrupted;

      if ( isInside && this.options.upInside ) {
        this.options.upInside( event, this.downTrail );
      }
      else if ( !isInside && this.options.upOutside ) {
        this.options.upOutside( event, this.downTrail );
      }
    }

    if ( this.options.up ) {
      this.options.up( event, this.downTrail );
    }
    event.currentTarget = currentTargetSave; // be polite to other listeners, restore currentTarget

    sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
  }

  /*---------------------------------------------------------------------------*
   * events called from the node input listener
   *----------------------------------------------------------------------------*/

  /**
   * mouse/touch down on this node
   * @public (scenery-internal)
   *
   * @param {SceneryEvent} event
   */
  down( event ) {
    this.buttonDown( event );
  }

  /**
   * Called when input is interrupted on this listener, see https://github.com/phetsims/scenery/issues/218
   * @public
   */
  interrupt() {
    if ( this.isDown ) {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'DownUpListener interrupt' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      this.interrupted = true;

      const context = EventContext.createSynthetic();

      // We create a synthetic event here, as there is no available event here.
      // Empty trail, so that it for-sure isn't under our downTrail (guaranteeing that isInside will be false).
      const syntheticEvent = new SceneryEvent( new Trail(), 'synthetic', this.pointer, context );
      syntheticEvent.currentTarget = this.downCurrentTarget;
      this.buttonUp( syntheticEvent );

      this.interrupted = false;

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    }
  }
}

scenery.register( 'DownUpListener', DownUpListener );

export default DownUpListener;