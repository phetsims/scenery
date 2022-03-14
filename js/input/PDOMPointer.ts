// Copyright 2018-2022, University of Colorado Boulder

/**
 * Pointer type for managing accessibility, in particular the focus in the display.
 * Tracks the state of accessible focus.
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import Vector2 from '../../../dot/js/Vector2.js';
import { scenery, Trail, Pointer, Focus, FocusManager, PDOMInstance, Display, Node } from '../imports.js';

export default class PDOMPointer extends Pointer {

  // (scenery-internal) - Prevent any "trusted" events from being dispatched to the KeyStateTracker. When
  // true, only scripted events are passed to the keyStateTracker. Otherwise, the modeled keyboard state when using
  // fuzzBoard will appear broken as both user and KeyboardFuzzer interact with display.
  blockTrustedEvents: boolean;

  private display: Display;

  // target of a user event, if focus changes in response to keydown listeners, listeners
  // on keyup are prevented because the key press was not intended for the newly focused node.
  // TODO: Can we do this for more than keydown/keyup? See https://github.com/phetsims/scenery/issues/942
  private keydownTargetNode: Node | null;

  constructor( display: Display ) {
    // We'll start with a defined Vector2, so that pointers always have points
    super( Vector2.ZERO, false, 'pdom' );

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
  private initializeListeners() {

    this.addInputListener( {
      focus: event => {
        assert && assert( this.trail, 'trail should have been calculated for the focused node' );

        const lastNode = this.trail!.lastNode();

        // NOTE: The "root" peer can't be focused (so it doesn't matter if it doesn't have a node).
        if ( lastNode.focusable ) {
          const visualTrail = PDOMInstance.guessVisualTrail( this.trail!, this.display.rootNode );

          FocusManager.pdomFocus = new Focus( this.display, visualTrail );
          this.point = visualTrail.parentToGlobalPoint( lastNode.center );
        }
        else {

          // It is possible that `blur` or `focusout` listeners have removed the element from the traversal order
          // before we receive the `focus` event. In that case, the browser will still try to put focus on the element
          // even though the PDOM element and Node are not in the traversal order. It is more consistent to remove
          // focus in this case.
          event.target.blur();

          // do not allow any more focus listeners to dispatch, this Node should never have been focused in the
          // first place, but the browser did it anyway
          event.abort();
        }
      },
      blur: event => {
        assert && assert( event.domEvent );

        // Null if it is not in the PDOM, or if it is undefined
        const relatedTargetTrail = this.display._input!.getRelatedTargetTrail( event.domEvent! );

        this.trail = null;

        if ( relatedTargetTrail && relatedTargetTrail.lastNode().focusable ) {
          FocusManager.pdomFocus = new Focus( this.display, PDOMInstance.guessVisualTrail( relatedTargetTrail, this.display.rootNode ) );
        }
        else {

          // Don't set this before the related target case because we want to support Node.blur listeners overwriting
          // the relatedTarget behavior.
          FocusManager.pdomFocus = null;
        }

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

  /**
   * @returns - updated trail
   */
  updateTrail( trailId: string ): Trail {
    if ( this.trail && this.trail.getUniqueId() === trailId ) {
      return this.trail;
    }
    const trail = Trail.fromUniqueId( this.display.rootNode, trailId );
    this.trail = trail;
    return trail;
  }

  /**
   * Recompute the trail to the node under this PDOMPointer. Updating the trail here is generally not necessary since
   * it is recomputed on focus. But there are times where pdom events can be called out of order with focus/blur
   * and the trail will either be null or stale. This might happen more often when scripting fake browser events
   * with a timeout (like in fuzzBoard).
   * (scenery-internal)
   */
  invalidateTrail( trailString: string ) {
    if ( this.trail === null || this.trail.uniqueId !== trailString ) {
      this.trail = Trail.fromUniqueId( this.display.rootNode, trailString );
    }
  }
}

scenery.register( 'PDOMPointer', PDOMPointer );
