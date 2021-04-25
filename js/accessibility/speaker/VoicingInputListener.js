// Copyright 2021, University of Colorado Boulder

/**
 * Trying out a single listener to be added to Display that will manage all input related to the voicing
 * feature. Will actually speak content when a Node that composes Voicing receives a down/click event. Also
 * updates the Display's pointerFocusProperty when Pointers move over Nodes that compose Voicing.
 *
 * A work in progress, don't use yet.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import scenery from '../../scenery.js';
import Focus from '../Focus.js';
import ReadingBlock from './ReadingBlock.js';
import voicingManager from './voicingManager.js';

class VoicingInputListener {

  /**
   * @param {BooleanProperty} voicingEnabledProperty
   * @param {Display} display
   */
  constructor( display, voicingEnabledProperty ) {

    // @private {Display}
    this.display = display;

    // @private {Trail|null} - reference to the active trail under DOM focus, updated in response to getting the trail
    // for those events
    this.activeFocusTrail = null;

    // @private {Trail|null} - references to the active trail under the mouse/touch pointer, used on down and over
    // to get responses for those events
    this.activePointerTrail = null;

    // @private {BooleanProperty}
    this.voicingEnabledProperty = voicingEnabledProperty;
  }

  /**
   * Called in response to a change of focus.
   * @public (part of the scenery listener API)
   *
   * @param event
   */
  focus( event ) {
    const voicingNode = this.findVoicingNode( event.trail );
    if ( voicingNode ) {
      this.activeFocusTrail = event.trail;

      // there is never a context response on focus
      this.speakVoicingContent( voicingNode, event );
    }
    else {
      this.activeFocusTrail = null;
    }
  }

  /**
   * Called in response to loss of focus.
   * @public
   * @param event
   */
  blur( event ) {
    this.activeFocusTrail = null;
  }

  /**
   * Called in response to a Pointer move event.
   * @public (part of the scenery listener API)
   *
   * @param {SceneryEvent} event
   */
  move( event ) {
    let hitNode = null;

    // only search for VoicingHitShapes if voicing is enabled
    if ( this.voicingEnabledProperty.value && this.display.readingBlockHighlightsVisibleProperty.value ) {
      ReadingBlock.ReadingBlockHitShapes.forEach( ( readingBlockHitShape, voicingNode ) => {
        const localToGlobalMatrix = voicingNode.getLocalToGlobalMatrix();

        const transformedHitShape = readingBlockHitShape.transformed( localToGlobalMatrix );
        if ( transformedHitShape.containsPoint( event.pointer.point ) ) {

          // now to a hit test on the Node itself to determine if it is would be hittable
          const transformedMouseArea = voicingNode.mouseArea.transformed( localToGlobalMatrix );
          const hitTrail = this.display.rootNode.hitTest( transformedMouseArea.bounds.center, true );
          if ( hitTrail && hitTrail.containsNode( voicingNode ) ) {
            hitNode = voicingNode;
          }
        }
      } );
    }

    // if we haven't found a hit yet, search for Nodes that compose Voicing that don't use a hit shape - these
    // will be interactive
    if ( !hitNode ) {

      // check for interactive Nodes that compose voicing
      for ( let i = event.trail.nodes.length - 1; i >= 0; i-- ) {
        const node = event.trail.nodes[ i ];
        if ( node.voicing ) {
          hitNode = node;
          break;
        }
      }
    }

    if ( hitNode ) {
      const uniqueTrail = hitNode.getUniqueTrail();
      if ( this.activePointerTrail === null || !uniqueTrail.equals( this.activePointerTrail ) ) {
        this.activePointerTrail = uniqueTrail;
        this.display.pointerFocusProperty.set( new Focus( this.display, uniqueTrail ) );
      }
    }
    else if ( this.activePointerTrail !== null ) {
      this.activePointerTrail = null;
      this.display.pointerFocusProperty.set( null );
    }
  }

  /**
   * Called in response to the down input event.
   * @public (part of scenery listener API)
   *
   * @param {SceneryEvent} event
   */
  down( event ) {
    this.respondToActivation( event );
  }

  /**
   * Called in response to the click input event.
   * @public (part of scenery listener API)
   *
   * @param {SceneryEvent} event
   */
  click( event ) {
    this.respondToActivation( event );
  }

  /**
   * Collect responses that would come from an activation event (pointer down or alternative input click).
   * @private
   */
  respondToActivation( event ) {
    const activeTrail = this.activePointerTrail || this.activeFocusTrail;
    if ( activeTrail ) {
      const voicingNode = activeTrail.lastNode();
      if ( voicingNode ) {
        this.speakVoicingContent( voicingNode, event );
      }
    }
  }

  /**
   * Speek the content from the VoicingNod in response to input.
   * @private
   *
   * @param {Node} voicingNode
   * @param {SceneryEvent} event
   */
  speakVoicingContent( voicingNode, event ) {
    const response = voicingManager.collectResponses( {
      objectResponse: voicingNode.voicingCreateObjectResponse( event ),
      interactionHint: voicingNode.voicingCreateHintResponse( event ),
      contextResponse: voicingNode.voicingCreateContextResponse( event ),
      overrideResponse: voicingNode.voicingCreateOverrideResponse( event )
    } );

    // don't send to utteranceQueue if response is empty
    if ( response ) {
      const utteranceQueue = voicingNode.utteranceQueue || this.display.voicingUtteranceQueue;
      utteranceQueue.addToBack( response );
    }
  }

  /**
   * Given a trail, find the Node that composes Voicing.
   * @private
   *
   * @param {Trail} trail
   * @returns {null|Node}
   */
  findVoicingNode( trail ) {
    let voicingNode = null;
    for ( let i = 0; i < trail.length; i++ ) {
      if ( trail.nodes[ i ].voicing ) {
        voicingNode = trail.nodes[ i ];
      }
    }

    return voicingNode;
  }

  /**
   * @private
   * @param voicingNode
   */
  speakVoicingNodeContent( voicingNode ) {

  }
}

scenery.register( 'VoicingInputListener', VoicingInputListener );
export default VoicingInputListener;
