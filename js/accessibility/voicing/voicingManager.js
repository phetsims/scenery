// Copyright 2021, University of Colorado Boulder

/**
 * Manages output of responses for the Voicing feature. First, see Voicing.js for a description of what that includes.
 * This singleton is responsible for controlling when responses of each category are spoken when speech is
 * requested for a Node composed with Voicing.
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

    // @public {BooleanProperty} - whether or not component names are read as input lands on various components
    this.nameResponsesEnabledProperty = new BooleanProperty( true );

    // @public {BooleanProperty} - whether or not "Object Responses" are read as interactive components change
    this.objectResponsesEnabledProperty = new BooleanProperty( true );

    // @public {BooleanProperty} - whether or not "Context Responses" are read as inputs receive interaction
    this.contextResponsesEnabledProperty = new BooleanProperty( true );

    // @public {BooleanProperty} - whether or not "Hints" are read to the user in response to certain input
    this.hintResponsesEnabledProperty = new BooleanProperty( false );

    // @public {BooleanProperty} - Controls whether Voicing is enabled in a "main window" area of the application.
    // This supports the ability to disable Voicing for the important screen content of your simulation while keeping
    // Voicing for surrounding UI components enabled. At the time of this writing, all "Reading Blocks" are disabled
    // when Voicing for the "main window" is disabled. See ReadingBlock.js for more information.
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

      // {boolean} - if true, the nameResponse, objectResponse, contextResponse, and interactionHint will all be spoken
      // regardless of the values of the Properties of voicingManager
      ignoreProperties: false,

      // {Object} - The collection of string patterns to use when assembling responses based on which
      // responses are provided and which voicingManager Properties are true. See VoicingResponsePatterns
      // if you do not want to use the default.
      responsePatterns: VoicingResponsePatterns.DEFAULT_RESPONSE_PATTERNS
    }, options );

    VoicingResponsePatterns.validatePatternKeys( options.responsePatterns );

    const usesNames = options.nameResponse && ( this.nameResponsesEnabledProperty.get() || options.ignoreProperties );
    const usesObjectChanges = options.objectResponse && ( this.objectResponsesEnabledProperty.get() || options.ignoreProperties );
    const usesContextChanges = options.contextResponse && ( this.contextResponsesEnabledProperty.get() || options.ignoreProperties );
    const usesInteractionHints = options.hintResponse && ( this.hintResponsesEnabledProperty.get() || options.ignoreProperties );
    const responseKey = VoicingResponsePatterns.createPatternKey( usesNames, usesObjectChanges, usesContextChanges, usesInteractionHints );

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
