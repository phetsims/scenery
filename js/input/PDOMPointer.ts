// Copyright 2018-2023, University of Colorado Boulder

/**
 * Pointer type for managing accessibility, in particular the focus in the display.
 * Tracks the state of accessible focus.
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import Vector2 from '../../../dot/js/Vector2.js';
import { Display, Node, PDOMInstance, Pointer, scenery, Trail } from '../imports.js';

export default class PDOMPointer extends Pointer {

  // (scenery-internal) - Prevent any "trusted" events from being dispatched to the KeyStateTracker. When
  // true, only scripted events are passed to the keyStateTracker. Otherwise, the modeled keyboard state when using
  // fuzzBoard will appear broken as both user and KeyboardFuzzer interact with display.
  public blockTrustedEvents: boolean;

  private readonly display: Display;

  // target of a user event, if focus changes in response to keydown listeners, listeners
  // on keyup are prevented because the key press was not intended for the newly focused node.
  // We only do this for keyup/keydown because focus can change between them, but it is not necessary
  // for other single events like 'click', 'input' or 'change. See https://github.com/phetsims/scenery/issues/942
  private keydownTargetNode: Node | null;

  public constructor( display: Display ) {
    // We'll start with a defined Vector2, so that pointers always have points
    super( Vector2.ZERO, 'pdom' );

    this.display = display;

    this.initializeListeners();

    this.blockTrustedEvents = false;

    this.keydownTargetNode = null;

    sceneryLog && sceneryLog.Pointer && sceneryLog.Pointer( `Created ${this.toString()}` );
  }

  /**
   * Set up listeners, attaching blur and focus listeners to the pointer once this PDOMPointer has been attached
   * to a display.
   */
  private initializeListeners(): void {

    this.addInputListener( {
      focus: event => {
        assert && assert( this.trail, 'trail should have been calculated for the focused node' );

        const lastNode = this.trail!.lastNode();

        // NOTE: The "root" peer can't be focused (so it doesn't matter if it doesn't have a node).
        if ( lastNode.focusable ) {
          const visualTrail = PDOMInstance.guessVisualTrail( this.trail!, this.display.rootNode );
          this.point = visualTrail.parentToGlobalPoint( lastNode.center );

          // TODO: it would be better if we could use this assertion instead, but guessVisualTrail seems to not be working here, https://github.com/phetsims/phet-io/issues/1847
          if ( isNaN( this.point.x ) ) {
            this.point.setXY( 0, 0 );
            // assert && assert( !isNaN( this.point.x ), 'Guess visual trail should be able to get the right point' );
          }
        }
      },
      blur: event => {
        this.trail = null;
        this.keydownTargetNode = null;
      },
      keydown: event => {
        if ( this.blockTrustedEvents && event.domEvent!.isTrusted ) {
          return;
        }

        // set the target to potentially block keyup events
        this.keydownTargetNode = event.target;
      },
      keyup: event => {
        if ( this.blockTrustedEvents && event.domEvent!.isTrusted ) {
          return;
        }

        // The keyup event was received on a node that didn't receive a keydown event, abort to prevent any other
        // listeners from being called for this event. Done after updating KeyStateTracker so that the global state
        // of the keyboard is still accurate
        if ( this.keydownTargetNode !== event.target ) {
          event.abort();
        }
      }
    } );
  }

  public updateTrail( trail: Trail ): Trail {

    // overwrite this.trail if we don't have a trail yet, or if the new trail doesn't equal the old one.
    if ( !( this.trail && this.trail.equals( trail ) ) ) {
      this.trail = trail;
    }
    return this.trail;
  }
}

scenery.register( 'PDOMPointer', PDOMPointer );
