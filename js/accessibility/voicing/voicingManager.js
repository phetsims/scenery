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

import BooleanProperty from '../../../../axon/js/BooleanProperty.js';
import merge from '../../../../phet-core/js/merge.js';
import StringUtils from '../../../../phetcommon/js/util/StringUtils.js';
import scenery from '../../scenery.js';
import VoicingResponsePatterns from './VoicingResponsePatterns.js';

class VoicingManager {
  constructor() {

    // @public {BooleanProperty} - whether or not object names are read as input lands on various components
    this.namesProperty = new BooleanProperty( true );

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

      // {string|null} - spoken when name responses are enabled
      nameResponse: null,

      // {string|null} - spoken when object responses are enabled
      objectResponse: null,

      // {string|null} - spoken when context responses are enabled
      contextResponse: null,

      // {string|null} - spoken when interaction hints are enabled
      hintResponse: null,

      // {boolean} - if true, the objectResponse, contextResponse, and interactionHint will all be spoken
      // regardless of the values of the Properties of voicingManager
      ignoreProperties: false,

      // {Object} - The collection of string patterns to use when assembling responses based on which
      // responses are provided and which voicingManager Properties are true. See VoicingResponsePatterns
      // if you do not want to use the default.
      responsePatterns: VoicingResponsePatterns.DEFAULT_RESPONSE_PATTERNS
    }, options );

    VoicingResponsePatterns.validatePatternKeys( options.responsePatterns );

    const usesNames = options.nameResponse && ( this.namesProperty.get() || options.ignoreProperties );
    const usesObjectChanges = options.objectResponse && ( this.objectChangesProperty.get() || options.ignoreProperties );
    const usesContextChanges = options.contextResponse && ( this.contextChangesProperty.get() || options.ignoreProperties );
    const usesInteractionHints = options.hintResponse && ( this.hintsProperty.get() || options.ignoreProperties );

    // generate the key to find the string pattern to use from options.responsePatterns
    let responses = '';
    if ( usesNames ) { responses = responses.concat( 'NAME'.concat( '_' ) ); }
    if ( usesObjectChanges ) { responses = responses.concat( 'OBJECT'.concat( '_' ) ); }
    if ( usesContextChanges ) { responses = responses.concat( 'CONTEXT'.concat( '_' ) ); }
    if ( usesInteractionHints ) { responses = responses.concat( 'HINT'.concat( '_' ) ); }
    const responseKey = _.camelCase( responses );

    let finalResponse = '';
    if ( responseKey ) {

      // graceful if the responseKey is empty, but if we formed some key, it should
      // be defined in responsePatterns
      const patternString = options.responsePatterns[ responseKey ];
      assert && assert( patternString, `no pattern string found for key ${responseKey}` );

      finalResponse = StringUtils.fillIn( patternString, {
        NAME: options.nameResponse,
        OBJECT: options.objectResponse,
        CONTEXT: options.contextResponse,
        HINT: options.hintResponse
      } );
    }

    return finalResponse;
  }
}

const voicingManager = new VoicingManager();

scenery.register( 'voicingManager', voicingManager );
export default voicingManager;
