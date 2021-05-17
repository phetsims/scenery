// Copyright 2021, University of Colorado Boulder

/**
 * A trait for the Voicing feature that can be composed with a Node. Allows you to specify callbacks that generate
 * responses that are spoken with the speech synthesis, highlights that are only displayed when voicing is enabled,
 * and other attributes of the feature at the Node level.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import extend from '../../../../phet-core/js/extend.js';
import inheritance from '../../../../phet-core/js/inheritance.js';
import merge from '../../../../phet-core/js/merge.js';
import Node from '../../nodes/Node.js';
import scenery from '../../scenery.js';
import MouseHighlighting from './MouseHighlighting.js';
import voicingManager from './voicingManager.js';
import VoicingResponsePatterns from './VoicingResponsePatterns.js';
import voicingUtteranceQueue from './voicingUtteranceQueue.js';

// options that are supported by Voicing.js. Added to mutator keys so that Voicing properties can be set with mutate.
const VOICING_OPTION_KEYS = [
  'voicingNameResponse',
  'voicingObjectResponse',
  'voicingContextResponse',
  'voicingHintResponse',
  'voicingUtteranceQueue'
];

const Voicing = {
  compose( type ) {
    assert && assert( _.includes( inheritance( type ), Node ), 'Only Node subtypes should compose Voicing' );

    const proto = type.prototype;

    // compose with mouse highlighting
    MouseHighlighting.compose( type );

    extend( proto, {

      /**
       * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in
       * the order they will be evaluated.
       * @protected
       *
       * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
       *       cases that may apply.
       */
      _mutatorKeys: VOICING_OPTION_KEYS.concat( proto._mutatorKeys ),

      /**
       * Initialize in the type being composed with Voicing. Call this in the constructor.
       * @param {Object} [options] - NOTE: much of the time, the Node this composes into will call mutate for you, be careful not to double call.
       * @public
       */
      initializeVoicing( options ) {

        assert && assert( this.voicing === undefined, 'Voicing has already been initialized for this Node' );

        // initialize "super" Trait to support highlights on mouse input
        this.initializeMouseHighlighting();

        // @public (read-only) - flag indicating that this Node is composed with Voicing functionality
        this.voicing = true;

        // @private {string|null} - The response to be spoken for this Node when speaking names. This is usually
        // the accessible name for the Node, typically spoken on focus and on interaction, labelling what the object is.
        this._voicingNameResponse = null;

        // @private {string|null} - The response to be spoken for this node when speaking object changes. This is
        // usually the response describing what changes about an object in response to interaction.
        this._voicingObjectResponse = null;

        // @private {string|null} - The response to be spoken for this node when speaking about context changes.
        // This is usually a response that describes the contextual changes that have occurred after interacting
        // with the object.
        this._voicingContextResponse = null;

        // @private {string|null} - The response to be spoken when speaking hints. This is usually the response
        // that guides the user toward further interaction with this object if it is important to do so to use
        // the application.
        this._voicingHintResponse = null;

        // @private {boolean} - Controls whether or not object, context, and hint responses are controlled
        // by voicingManager Properties. If true, all responses will be spoken when requested, regardless
        // of these Properties. This is often useful for surrounding UI components where it is important
        // that information be heard even when certain responses have been disabled.
        this._voicingIgnoreVoicingManagerProperties = false;

        // @private {UtteranceQueue} - The utteranceQueue that responses for this Node will be spoken through.
        // By default, it will go through the singleton voicingUtteranceQueue, but you may need separate
        // UtteranceQueues for different areas of content in your application.
        this._voicingUtteranceQueue = null;

        // {Object} - A collection of response patterns that are used to collect the responses of this Voicing Node
        // with voicingManager. See VoicingResponsePatterns for more details.
        this._voicingResponsePatterns = VoicingResponsePatterns.DEFAULT_RESPONSE_PATTERNS;

        // @private {Object} - Input listener that speaks content on focus. This is the only input listener added
        // by Voicing, but it is the one that is consistent for all Voicing nodes. On focus, speak the name, object
        // response, and interaction hint.
        this.speakContentOnFocusListener = {
          focus: event => {
            this.voicingSpeakFullResponse( {
              contextResponse: null
            } );
          }
        };
        this.addInputListener( this.speakContentOnFocusListener );

        if ( options ) {
          this.mutate( _.pick( options, VOICING_OPTION_KEYS ) );
        }
      },

      /**
       * Speak all responses assigned to this Node. Options allow you to override a response for the particular case,
       * or assign an Utterance to control the flow of this response.
       *
       * to tap into the responses you want heard.
       * @public
       *
       * @param {Object} [options]
       */
      voicingSpeakFullResponse( options ) {
        options = merge( {
          nameResponse: this._voicingNameResponse,
          objectResponse: this._voicingObjectResponse,
          contextResponse: this._voicingContextResponse,
          hintResponse: this._voicingHintResponse
        }, options );

        this.collectAndSpeakResponse( options );
      },

      /**
       * Speak the provided responses that you pass in with options. Note that this will NOT speak the name, object,
       * context, or hint responses assigned to this node by default. If you want to speak the responses assigned
       * to this Node, use voicingSpeakFullResponse.
       * @public
       *
       * @param {Object} [options]
       */
      voicingSpeakResponse( options ) {
        this.collectAndSpeakResponse( options );
      },

      /**
       * Speak only the name response assigned to this Node.
       * @param options
       */
      voicingSpeakNameResponse( options ) {
        options = merge( {
          nameResponse: this._voicingNameResponse
        }, options );

        this.collectAndSpeakResponse( options );
      },

      /**
       * Speak only the object response assigned to this Node.
       * @public
       *
       * @param {Object} [options]
       */
      voicingSpeakObjectResponse( options ) {
        options = merge( {
          objectResponse: this._voicingObjectResponse
        }, options );

        this.collectAndSpeakResponse( options );
      },

      /**
       * Speak only the context response assigned to this Node.
       * @public
       *
       * @param {Object} [options]
       */
      voicingSpeakContextResponse( options ) {
        options = merge( {
          contextResponse: this._voicingContextResponse
        }, options );

        this.collectAndSpeakResponse( options );
      },


      /**
       * Speak only the hint response assigned to this Node.
       * @public
       *
       * @param {Object} [options]
       */
      voicingSpeakHintResponse( options ) {
        options = merge( {
          hintResponse: this._voicingHintResponse
        }, options );

        this.collectAndSpeakResponse( options );
      },

      /**
       * Collect responses with the voicingManager and speak the output with an UtteranceQueue.
       * @param options
       */
      collectAndSpeakResponse( options ) {
        options = merge( {

          // {boolean} - whether or not this response should ignore the Properties of voicingManager
          ignoreProperties: this._ignoreVoicingManagerProperties,

          // {Object} - collection of string patterns to use with voicingManager.collectResponses, see
          // VoicingResponsePatterns for more information.
          responsePatterns: this._voicingResponsePatterns,

          // {Utterance|null} - The utterance to use if you want this response to be more controlled in the
          // UtteranceQueue.
          utterance: null
        }, options );

        const response = voicingManager.collectResponses( options );
        console.log( response );

        if ( options.utterance ) {
          options.utterance.alert = response;
          this.speakContent( options.utterance );
        }
        else {
          this.speakContent( response );
        }
      },

      /**
       * Sets the voicingNameResponse for this Node. This is usually the label of the element and is spoken
       * when the object receives input.
       * @public
       *
       * @param {string|null} response
       */
      setVoicingNameResponse( response ) {
        this._voicingNameResponse = response;
      },
      set voicingNameResponse( response ) { this.setVoicingNameResponse( response ); },

      /**
       * Get the voicingNameResponse for this Node.
       * @public
       *
       * @returns {string|null}
       */
      getVoicingNameResponse() {
        return this._voicingNameResponse;
      },
      get voicingNameResponse() { return this.getVoicingNameResponse(); },

      /**
       * Set the object response for this Node.
       * @public
       *
       * @param {string|null} response
       */
      setVoicingObjectResponse( response ) {
        this._voicingObjectResponse = response;
      },
      set voicingObjectResponse( response ) { this.setVoicingObjectResponse( response ); },

      /**
       * Gets the object response for this Node.
       * @public
       *
       * @returns {string}
       */
      getVoicingObjectResponse() {
        return this._voicingObjectResponse;
      },
      get voicingObjectResponse() { return this.getVoicingObjectResponse(); },

      /**
       * Set the context response for this Node.
       * @public
       *
       * @param {string|null} response
       */
      setVoicingContextResponse( response ) {
        this._voicingContextResponse = response;
      },
      set voicingContextResponse( response ) { this.setVoicingContextResponse( response ); },

      /**
       * Gets the context response for this Node.
       * @public
       *
       * @returns {string|null}
       */
      getVoicingContextResponse() {
        return this._voicingContextResponse;
      },
      get voicingContextResponse() { return this.getVoicingContextResponse(); },

      /**
       * Sets the hint response for this Node.
       * @public
       *
       * @param {string|null} response
       */
      setVoicingHintResponse( response ) {
        this._voicingHintResponse = response;
      },
      set voicingHintResponse( response ) { this.setVoicingHintResponse( response ); },

      /**
       * Gets the hint response for this Node.
       * @public
       *
       * @returns {string|null}
       */
      getVoicingHintResponse() {
        return this._voicingHintResponse;
      },
      get voicingHintResponse() { return this.getVoicingHintResponse(); },

      /**
       * Set whether or not all responses for this Node will ignore the Properties of voicingManager. If false,
       * all responses will be spoken regardless of voicingManager Properties, which are generally set in user
       * preferences.
       * @public
       */
      setVoicingIgnoreVoicingManagerProperties( ignoreProperties ) {
        this._voicingIgnoreVoicingManagerProperties = ignoreProperties;
      },
      set voicingIgnoreVoicingManagerProperties( ignoreProperties ) { this.setVoicingIgnoreVoicingManagerProperties( ignoreProperties ); },

      /**
       * Get whether or not responses are ignoring voicingManager Properties.
       */
      getVoicingIgnoreVoicingManagerProperties() {
        return this._voicingIgnoreVoicingManagerProperties;
      },
      get voicingIgnoreVoicingManagerProperties() { return this.getVoicingIgnoreVoicingManagerProperties(); },

      /**
       * Sets the collection of patterns to use for voicing responses, controlling the order, punctuation, and
       * additional content for each combination of response. See VoicingResponsePatterns.js if you wish to use
       * a collection of string patterns that are not the default.
       * @public
       *
       * @param {Object} patterns - see VoicingResponsePatterns
       */
      setVoicingResponsePatterns( patterns ) {
        this._voicingResponsePatterns = patterns;
      },
      set voicingResponsePatterns( patterns ) { this.setVoicingResponsePatterns( patterns ); },

      /**
       * Get the VoicingResponsePatterns object that this Voicing Node is using to collect responses.
       * @public
       *
       * @returns {Object}
       */
      getVoicingResponsePatterns() {
        return this._voicingResponsePatterns;
      },
      get voicingResponsePatterns() { return this.getVoicingResponsePatterns(); },

      /**
       * Sets the utteranceQueue through which voicing associated with this Node will be spoken. By default,
       * the Display's voicingUtteranceQueue is used. But you can specify a different one if more complicated
       * management of voicing is necessary.
       * @public
       *
       * @param {UtteranceQueue} utteranceQueue
       */
      setVoicingUtteranceQueue( utteranceQueue ) {
        this._voicingUtteranceQueue = utteranceQueue;
      },
      set utteranceQueue( utteranceQueue ) { this.setVoicingUtteranceQueue( utteranceQueue ); },

      /**
       * Gets the utteranceQueue through which voicing associated with this Node will be spoken.
       * @public
       *
       * @returns {UtteranceQueue}
       */
      getUtteranceQueue() {
        return this._voicingUtteranceQueue;
      },
      get utteranceQueue() { return this.getUtteranceQueue(); },

      /**
       * Detaches references that ensure this components of this Trait are eligible for garbage collection.
       * @public
       */
      disposeVoicing() {
        this.removeInputListener( this.speakContentOnFocusListener );
        this.disposeMouseHighlighting();
      },

      /**
       * Use the provided function to create content to speak in response to Input. The content then added to the
       * back of the voicing utterance queue.
       * @protected
       *
       * @param {null|AlertableDef} content
       */
      speakContent( content ) {

        // don't send to utteranceQueue if response is empty
        if ( content ) {
          const utteranceQueue = this.utteranceQueue || voicingUtteranceQueue;
          utteranceQueue.addToBack( content );
        }
      }
    } );
  }
};

scenery.register( 'Voicing', Voicing );
export default Voicing;
