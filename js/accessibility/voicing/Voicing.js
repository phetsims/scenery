// Copyright 2021, University of Colorado Boulder

/**
 * A trait for Node that supports the Voicing feature, under accessibility. Allows you to define responses for the Node
 * and make requests to speak that content using HTML5 SpeechSynthesis and the UtteranceQueue. Voicing content is
 * organized into four categories which are responsible for describing different things. Output of this content
 * can be controlled by the responseCollector. These include the
 *
 * - "Name" response: The name of the object that uses Voicing. Similar to the "Accessible Name" in web accessibility.
 * - "Object" response: The state information about the object that uses Voicing.
 * - "Context" response: The contextual changes that result from interaction with the Node that uses Voicing.
 * - "Hint" response: A supporting hint that guides the user toward a desired interaction with this Node.
 *
 * See the property and setter documentation for each of these responses for more information.
 *
 * Once this content is set, you can make a request to speak it using an UtteranceQueue with one of the provided
 * functions in this Trait. It is up to you to call one of these functions when you wish for speech to be made. The only
 * exception is on the 'focus' event. Every Node that composes Voicing will speak its responses by when it
 * receives focus.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import extend from '../../../../phet-core/js/extend.js';
import inheritance from '../../../../phet-core/js/inheritance.js';
import merge from '../../../../phet-core/js/merge.js';
import Node from '../../nodes/Node.js';
import scenery from '../../scenery.js';
import MouseHighlighting from './MouseHighlighting.js';
import responseCollector from '../../../../utterance-queue/js/responseCollector.js';
import ResponsePatterns from '../../../../utterance-queue/js/ResponsePatterns.js';
import voicingUtteranceQueue from './voicingUtteranceQueue.js';

// options that are supported by Voicing.js. Added to mutator keys so that Voicing properties can be set with mutate.
const VOICING_OPTION_KEYS = [
  'voicingNameResponse',
  'voicingObjectResponse',
  'voicingContextResponse',
  'voicingHintResponse',
  'voicingUtteranceQueue',
  'voicingResponsePatterns',
  'voicingIgnoreVoicingManagerProperties'
];

const Voicing = {

  /**
   * @public
   * @trait {Node}
   * @mixes {MouseHighlighting}
   * @param {function(new:Node)} type - The type (constructor) whose prototype that is modified. Should be a Node class.
   */
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

        assert && assert( this.voicingInitialized === undefined, 'Voicing has already been initialized for this Node' );

        // initialize "super" Trait to support highlights on mouse input
        this.initializeMouseHighlighting( options );

        // @private {boolean} - to make sure that initializeVoicing is called before trying to use the mixin.
        this.voicingInitialized = true;

        // @private {string|null} - The response to be spoken for this Node when speaking names. This is usually
        // the accessible name for the Node, typically spoken on focus and on interaction, labelling what the object is.
        this._voicingNameResponse = null;

        // @private {string|null} - The response to be spoken for this node when speaking about object changes. This
        // is usually the state information directly associated with this Node, such as its current input value.
        this._voicingObjectResponse = null;

        // @private {string|null} - The response to be spoken for this node when speaking about context changes.
        // This is usually a response that describes the surrounding changes that have occurred after interacting
        // with the object.
        this._voicingContextResponse = null;

        // @private {string|null} - The response to be spoken when speaking hints. This is usually the response
        // that guides the user toward further interaction with this object if it is important to do so to use
        // the application.
        this._voicingHintResponse = null;

        // @private {boolean} - Controls whether or not name, object, context, and hint responses are controlled
        // by responseCollector Properties. If true, all responses will be spoken when requested, regardless
        // of these Properties. This is often useful for surrounding UI components where it is important
        // that information be heard even when certain responses have been disabled.
        this._voicingIgnoreVoicingManagerProperties = false;

        // @private {UtteranceQueue|null} - The utteranceQueue that responses for this Node will be spoken through.
        // By default (null), it will go through the singleton voicingUtteranceQueue, but you may need separate
        // UtteranceQueues for different areas of content in your application. For example, Voicing and
        // the default voicingUtteranceQueue may be disabled, but you could still want some speech to come through
        // while user is changing preferences or other settings.
        this._voicingUtteranceQueue = null;

        // {Object} - A collection of response patterns that are used to collect the responses of this Voicing Node
        // with responseCollector. Controls the order of the Voicing responses and even punctuation used when responses
        // are assembled into final content for the UtteranceQueue. See ResponsePatterns for more details.
        this._voicingResponsePatterns = ResponsePatterns.DEFAULT_RESPONSE_PATTERNS;

        // @private {Object} - Input listener that speaks content on focus. This is the only input listener added
        // by Voicing, but it is the one that is consistent for all Voicing nodes. On focus, speak the name, object
        // response, and interaction hint.
        this.speakContentOnFocusListener = {
          focus: () => {
            this.voicingSpeakFullResponse( {
              contextResponse: null
            } );
          }
        };
        this.addInputListener( this.speakContentOnFocusListener );

        // support passing options through directly on initialize
        if ( options ) {
          this.mutate( _.pick( options, VOICING_OPTION_KEYS ) );
        }
      },

      /**
       * Speak all responses assigned to this Node. Options allow you to override a responses for this particular
       * speech request. Each response is only spoken if the associated Property of responseCollector is true. If
       * all are Properties are false, nothing will be spoken.
       * @public
       *
       * @param {Object} [options]
       */
      voicingSpeakFullResponse( options ) {
        assert && assert( this.voicingInitialized, 'voicing must be initialized to speak' );

        // options are passed along to collectAndSpeakResponse, see that function for additional options
        options = merge( {
          nameResponse: this._voicingNameResponse,
          objectResponse: this._voicingObjectResponse,
          contextResponse: this._voicingContextResponse,
          hintResponse: this._voicingHintResponse
        }, options );

        this.collectAndSpeakResponse( options );
      },

      /**
       * Speak ONLY the provided responses that you pass in with options. This will NOT speak the name, object,
       * context, or hint responses assigned to this node by default. But it allows for clarity at usages so it is
       * clear that you are only requesting certain responses. If you want to speak all of the responses assigned
       * to this Node, use voicingSpeakFullResponse().
       *
       * Each response will only be spoken if the Properties of responseCollector are true. If all of those are false,
       * nothing will be spoken.
       * @public
       *
       * @param {Object} [options]
       */
      voicingSpeakResponse( options ) {
        assert && assert( this.voicingInitialized, 'voicing must be initialized to speak' );

        // options are passed along to collectAndSpeakResponse, see that function for additional options
        options = merge( {
          nameResponse: null,
          objectResponse: null,
          contextResponse: null,
          hintResponse: null
        }, options );

        this.collectAndSpeakResponse( options );
      },

      /**
       * By default, speak the name response. But accepts all other responses through options. Respects responseCollector
       * Properties, so the name response may not be spoken if responseCollector.nameResponseEnabledProperty is false.
       * @public
       *
       * @param {Object} [options]
       */
      voicingSpeakNameResponse( options ) {
        assert && assert( this.voicingInitialized, 'voicing must be initialized to speak' );

        // options are passed along to collectAndSpeakResponse, see that function for additional options
        options = merge( {
          nameResponse: this._voicingNameResponse
        }, options );

        this.collectAndSpeakResponse( options );
      },

      /**
       * By default, speak the object response. But accepts all other responses through options. Respects responseCollector
       * Properties, so the name response may not be spoken if responseCollector.objectResponseEnabledProperty is false.
       * @public
       *
       * @param {Object} [options]
       */
      voicingSpeakObjectResponse( options ) {
        assert && assert( this.voicingInitialized, 'voicing must be initialized to speak' );

        // options are passed along to collectAndSpeakResponse, see that function for additional options
        options = merge( {
          objectResponse: this._voicingObjectResponse
        }, options );

        this.collectAndSpeakResponse( options );
      },

      /**
       * By default, speak the context response. But accepts all other responses through options. Respects
       * responseCollector Properties, so the name response may not be spoken if
       * responseCollector.contextResponseEnabledProperty is false.
       * @public
       *
       * @param {Object} [options]
       */
      voicingSpeakContextResponse( options ) {
        assert && assert( this.voicingInitialized, 'voicing must be initialized to speak' );

        // options are passed along to collectAndSpeakResponse, see that function for additional options
        options = merge( {
          contextResponse: this._voicingContextResponse
        }, options );

        this.collectAndSpeakResponse( options );
      },

      /**
       * By default, speak the hint response. But accepts all other responses through options. Respects
       * responseCollector Properties, so the hint response may not be spoken if
       * responseCollector.hintResponseEnabledProperty is false.
       * @public
       *
       * @param {Object} [options]
       */
      voicingSpeakHintResponse( options ) {
        assert && assert( this.voicingInitialized, 'voicing must be initialized to speak' );

        // options are passed along to collectAndSpeakResponse, see that function for additional options
        options = merge( {
          hintResponse: this._voicingHintResponse
        }, options );

        this.collectAndSpeakResponse( options );
      },

      /**
       * Collect responses with the responseCollector and speak the output with an UtteranceQueue.
       * @protected
       *
       * @param {Object} [options]
       */
      collectAndSpeakResponse( options ) {
        options = merge( {

          // {boolean} - whether or not this response should ignore the Properties of responseCollector
          ignoreProperties: this._voicingIgnoreVoicingManagerProperties,

          // {Object} - collection of string patterns to use with responseCollector.collectResponses, see
          // ResponsePatterns for more information.
          responsePatterns: this._voicingResponsePatterns,

          // {Utterance|null} - The utterance to use if you want this response to be more controlled in the
          // UtteranceQueue.
          utterance: null
        }, options );

        let response = responseCollector.collectResponses( options );

        if ( options.utterance ) {
          options.utterance.alert = response;
          response = options.utterance;
        }
        this.speakContent( response );
      },

      /**
       * Use the provided function to create content to speak in response to input. The content is then added to the
       * back of the voicing UtteranceQueue.
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
      },

      /**
       * Sets the voicingNameResponse for this Node. This is usually the label of the element and is spoken
       * when the object receives input. When requesting speech, this will only be spoken if
       * responseCollector.nameResponsesEnabledProperty is set to true.
       *
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
       * Set the object response for this Node. This is usually the state information associated with this Node, such
       * as its current input value. When requesting speech, this will only be heard when
       * responseCollector.objectResponsesEnabledProperty is set to true.
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
       * @returns {string|null}
       */
      getVoicingObjectResponse() {
        return this._voicingObjectResponse;
      },
      get voicingObjectResponse() { return this.getVoicingObjectResponse(); },

      /**
       * Set the context response for this Node. This is usually the content that describes what has happened in
       * the surrounding application in response to interaction with this Node. When requesting speech, this will
       * only be heard if responseCollector.contextResponsesEnabledProperty is set to true.
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
       * Sets the hint response for this Node. This is usually a response that describes how to interact with this Node.
       * When requesting speech, this will only be spoken when responseCollector.hintResponsesEnabledProperty is set to
       * true.
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
       * Set whether or not all responses for this Node will ignore the Properties of responseCollector. If false,
       * all responses will be spoken regardless of responseCollector Properties, which are generally set in user
       * preferences.
       * @public
       */
      setVoicingIgnoreVoicingManagerProperties( ignoreProperties ) {
        this._voicingIgnoreVoicingManagerProperties = ignoreProperties;
      },
      set voicingIgnoreVoicingManagerProperties( ignoreProperties ) { this.setVoicingIgnoreVoicingManagerProperties( ignoreProperties ); },

      /**
       * Get whether or not responses are ignoring responseCollector Properties.
       */
      getVoicingIgnoreVoicingManagerProperties() {
        return this._voicingIgnoreVoicingManagerProperties;
      },
      get voicingIgnoreVoicingManagerProperties() { return this.getVoicingIgnoreVoicingManagerProperties(); },

      /**
       * Sets the collection of patterns to use for voicing responses, controlling the order, punctuation, and
       * additional content for each combination of response. See ResponsePatterns.js if you wish to use
       * a collection of string patterns that are not the default.
       * @public
       *
       * @param {Object} patterns - see ResponsePatterns.js
       */
      setVoicingResponsePatterns( patterns ) {
        this._voicingResponsePatterns = patterns;
      },
      set voicingResponsePatterns( patterns ) { this.setVoicingResponsePatterns( patterns ); },

      /**
       * Get the ResponsePatterns object that this Voicing Node is using to collect responses.
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

      set voicingUtteranceQueue( utteranceQueue ) { this.setVoicingUtteranceQueue( utteranceQueue ); },

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
       * Whether or not a Node composes Voicing.
       * @public
       * @returns {boolean}
       */
      get isVoicing() {
        return true;
      },

      /**
       * Detaches references that ensure this components of this Trait are eligible for garbage collection.
       * @public
       */
      disposeVoicing() {
        this.voicingInitialized = false;
        this.removeInputListener( this.speakContentOnFocusListener );
        this.disposeMouseHighlighting();
      }
    } );
  }
};

scenery.register( 'Voicing', Voicing );
export default Voicing;
