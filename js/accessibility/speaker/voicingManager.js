// Copyright 2021, University of Colorado Boulder

/**
 * Manages "voicing" content as it is read with speech synthesis. Output is categorized into
 *    "Object Responses" - Speech describing the object as it receives interaction
 *    "Context Responses" - Speech describing surrounding contextual changes in response to user interaction
 *    "Hints" - General hint content to guide a particular user interaction
 *
 * Output from each of these categories can be separately enabled and disabled.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import merge from '../../../../phet-core/js/merge.js';
import scenery from '../../scenery.js';
import BooleanProperty from '../../../../axon/js/BooleanProperty.js';

class VoicingManager {
  constructor() {

    // @public {BooleanProperty} - whether or not "Object Responses" are read as interactive components change
    this.objectChangesProperty = new BooleanProperty( true );

    // @public {BooleanProperty} - whether or not "Context Responses" are read as simulation objects change
    this.contextChangesProperty = new BooleanProperty( true );

    // @public {BooleanProperty} - whether or not "Hints" are read to the user
    this.hintsProperty = new BooleanProperty( false );

    // @public {BooleanProperty} - controls whether Voicing is enabled in a "main window" area of the application.
    // This is different from whether voicing is enabled in general. This way it is possible to disable voicing
    // for the main content of the application while still allowing it to come through for surrounding application
    // controls.
    this.mainWindowVoicingEnabledProperty = new BooleanProperty( true );
  }

  /**
   * Prepares final output with an object response, a context response, and a hint. Each response
   * will only be added to the final string if that response type is included by the user. Rather than using
   * unique utterances, we use string interpolation so that the highlight around the abject being spoken
   * about stays lit for the entire combination of responses.
   * @public
   *
   * @param {Object} [options]
   */
  collectResponses( options ) {

    options = merge( {

      // {string|null} - spoken when object responses are enabled
      objectResponse: null,

      // {string|null} - spoken when context responses are enabled
      contextResponse: null,

      // {string|null} - spoken when interaction hints are enabled
      interactionHint: null,

      // {string|null} - If this is provided, it is the ONLY spoken string, and it is always spoken regardless of
      // speech output levels selected by the user as long as speech is enabled.
      overrideResponse: null
    }, options );

    const objectChanges = this.objectChangesProperty.get();
    const contextChanges = this.contextChangesProperty.get();
    const interactionHints = this.hintsProperty.get();

    let usedObjectString = '';
    let usedContextString = '';
    let usedInteractionHint = '';

    if ( objectChanges && options.objectResponse ) {
      usedObjectString = options.objectResponse;
    }
    if ( contextChanges && options.contextResponse ) {
      usedContextString = options.contextResponse;
    }
    if ( interactionHints && options.interactionHint ) {
      usedInteractionHint = options.interactionHint;
    }

    // used to combine with string literal, but we need to conditionally include punctuation so that
    // it isn't always read - it would be more clear if we had a number of string patterns and assembled
    // that way
    let outputString = '';
    if ( options.overrideResponse ) {
      outputString = options.overrideResponse;
    }
    else {
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
    }

    return outputString;
  }
}

const voicingManager = new VoicingManager();
scenery.register( 'voicingManager', voicingManager );
export default voicingManager;
