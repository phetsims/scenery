// Copyright 2021, University of Colorado Boulder

/**
 * Manages "self-voicing" content as it is read with speech synthesis. Output is categorized into
 *    "Object Responses" - Speech describing the object as it receives interaction
 *    "Context Responses" - Speech describing surrounding contextual changes in response to user interaction
 *    "Hints" - General hint content to guide a particular user interaction
 *
 * Output from each of these categories can be separately enabled and disabled.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import scenery from '../../scenery.js';
import BooleanProperty from '../../../../axon/js/BooleanProperty.js';

class SelfVoicingManager {
  constructor() {

    // @public {BooleanProperty} - whether or not "Object Responses" are read as interactive components change
    this.objectChangesProperty = new BooleanProperty( true );

    // @public {BooleanProperty} - whether or not "Context Responses" are read as simulation objects change
    this.contextChangesProperty = new BooleanProperty( true );

    // @public {BooleanProperty} - whether or not "Hints" are read to the user
    this.hintsProperty = new BooleanProperty( false );
  }

  /**
   * Prepares final output with an object response, a context response, and a hint. Each response
   * will only be added to the final string if that response type is included by the user. Rather than using
   * unique utterances, we use string interpolation so that the highlight around the abject being spoken
   * about stays lit for the entire combination of responses.
   * @public
   *
   * @param {string|undefined} [objectResponse]
   * @param {string|undefined} [contextResponse]
   * @param {string|undefined} [interactionHint]
   * @param {Object} [options]
   */
  collectResponses( objectResponse, contextResponse, interactionHint, options ) {
    const objectChanges = this.objectChangesProperty.get();
    const contextChanges = this.contextChangesProperty.get();
    const interactionHints = this.hintsProperty.get();
    let usedObjectString = '';
    let usedContextString = '';
    let usedInteractionHint = '';
    if ( objectChanges && objectResponse ) {
      usedObjectString = objectResponse;
    }
    if ( contextChanges && contextResponse ) {
      usedContextString = contextResponse;
    }
    if ( interactionHints && interactionHint ) {
      usedInteractionHint = interactionHint;
    }
    // used to combine with string literal, but we need to conditionally include punctuation so that
    // it isn't always read
    let outputString = '';
    if ( usedObjectString ) {
      outputString += usedObjectString;
    }
    if ( usedContextString ) {
      if ( outputString.length > 0 ) {
        outputString += ', ';
      }
      outputString = outputString + usedContextString;
    }
    if ( usedInteractionHint ) {
      if ( outputString.length > 0 ) {
        outputString += ', ';
      }
      outputString = outputString + usedInteractionHint;
    }
    return outputString;
  }
}

const selfVoicingManager = new SelfVoicingManager();
scenery.register( 'selfVoicingManager', selfVoicingManager );
export default selfVoicingManager;
