// Copyright 2021, University of Colorado Boulder

/**
 * A trait for Node that supports the Voicing feature, under accessibility. Allows you to define responses for the Node
 * and make requests to speak that content using HTML5 SpeechSynthesis and the UtteranceQueue. Voicing content is
 * organized into four categories which are responsible for describing different things. Responses are stored on the
 * composed type: "ResponsePacket." See that file for details about what responses it stores. Output of this content
 * can be controlled by the responseCollector. Responses are defined as the following. . .
 *
 * - "Name" response: The name of the object that uses Voicing. Similar to the "Accessible Name" in web accessibility.
 * - "Object" response: The state information about the object that uses Voicing.
 * - "Context" response: The contextual changes that result from interaction with the Node that uses Voicing.
 * - "Hint" response: A supporting hint that guides the user toward a desired interaction with this Node.
 *
 * See ResponsePacket, as well as the property and setter documentation for each of these responses for more
 * information.
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
import responseCollector from '../../../../utterance-queue/js/responseCollector.js';
import ResponsePacket from '../../../../utterance-queue/js/ResponsePacket.js';
import ResponsePatternCollection from '../../../../utterance-queue/js/ResponsePatternCollection.js';
import { InteractiveHighlighting, Node, scenery, voicingUtteranceQueue } from '../../imports.js';

// options that are supported by Voicing.js. Added to mutator keys so that Voicing properties can be set with mutate.
const VOICING_OPTION_KEYS = [
  'voicingNameResponse',
  'voicingObjectResponse',
  'voicingContextResponse',
  'voicingHintResponse',
  'voicingUtteranceQueue',
  'voicingResponsePatternCollection',
  'voicingIgnoreVoicingManagerProperties',
  'voicingFocusListener'
];

const Voicing = {

  /**
   * @public
   * @trait {Node}
   * @mixes {InteractiveHighlighting}
   * @param {function(new:Node)} type - The type (constructor) whose prototype that is modified. Should be a Node class.
   */
  compose( type ) {
    assert && assert( _.includes( inheritance( type ), Node ), 'Only Node subtypes should compose Voicing' );

    const proto = type.prototype;

    // compose with Interactive Highlights, all Nodes with Voicing features highlight as they are interactive
    InteractiveHighlighting.compose( type );

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

        // undefined OR support poolable with the value set by dispose
        assert && assert( this.voicingInitialized === undefined || this.voicingInitialized === false, 'Voicing has already been initialized for this Node' );

        // initialize "super" Trait to support highlights on mouse input
        this.initializeInteractiveHighlighting( options );

        // @private {boolean} - to make sure that initializeVoicing is called before trying to use the mixin.
        this.voicingInitialized = true;

        // @public {ResponsePacket} - ResponsePacket that holds all the supported responses to be Voiced
        this.voicingResponsePacket = new ResponsePacket();

        // @private {UtteranceQueue|null} - The utteranceQueue that responses for this Node will be spoken through.
        // By default (null), it will go through the singleton voicingUtteranceQueue, but you may need separate
        // UtteranceQueues for different areas of content in your application. For example, Voicing and
        // the default voicingUtteranceQueue may be disabled, but you could still want some speech to come through
        // while user is changing preferences or other settings.
        this._voicingUtteranceQueue = null;

        // @private {Function(event):} - called when this node is focused.
        this._voicingFocusListener = this.defaultFocusListener;

        // @private {Object} - Input listener that speaks content on focus. This is the only input listener added
        // by Voicing, but it is the one that is consistent for all Voicing nodes. On focus, speak the name, object
        // response, and interaction hint.
        this.speakContentOnFocusListener = {
          focus: event => {
            this._voicingFocusListener( event );
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
          nameResponse: this.voicingResponsePacket.nameResponse,
          objectResponse: this.voicingResponsePacket.objectResponse,
          contextResponse: this.voicingResponsePacket.contextResponse,
          hintResponse: this.voicingResponsePacket.hintResponse
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
          nameResponse: this.voicingResponsePacket.nameResponse
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
          objectResponse: this.voicingResponsePacket.objectResponse
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
          contextResponse: this.voicingResponsePacket.contextResponse
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
          hintResponse: this.voicingResponsePacket.hintResponse
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
          ignoreProperties: this.voicingResponsePacket.ignoreProperties,

          // {Object} - collection of string patterns to use with responseCollector.collectResponses, see
          // ResponsePatternCollection for more information.
          responsePatternCollection: this.voicingResponsePacket.responsePatternCollection,

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
          const utteranceQueue = this.voicingUtteranceQueue || voicingUtteranceQueue;
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
        this.voicingResponsePacket.nameResponse = response;
      },
      set voicingNameResponse( response ) { this.setVoicingNameResponse( response ); },

      /**
       * Get the voicingNameResponse for this Node.
       * @public
       *
       * @returns {string|null}
       */
      getVoicingNameResponse() {
        return this.voicingResponsePacket.nameResponse;
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
        this.voicingResponsePacket.objectResponse = response;
      },
      set voicingObjectResponse( response ) { this.setVoicingObjectResponse( response ); },

      /**
       * Gets the object response for this Node.
       * @public
       *
       * @returns {string|null}
       */
      getVoicingObjectResponse() {
        return this.voicingResponsePacket.objectResponse;
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
        this.voicingResponsePacket.contextResponse = response;
      },
      set voicingContextResponse( response ) { this.setVoicingContextResponse( response ); },

      /**
       * Gets the context response for this Node.
       * @public
       *
       * @returns {string|null}
       */
      getVoicingContextResponse() {
        return this.voicingResponsePacket.contextResponse;
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
        this.voicingResponsePacket.hintResponse = response;
      },
      set voicingHintResponse( response ) { this.setVoicingHintResponse( response ); },

      /**
       * Gets the hint response for this Node.
       * @public
       *
       * @returns {string|null}
       */
      getVoicingHintResponse() {
        return this.voicingResponsePacket.hintResponse;
      },
      get voicingHintResponse() { return this.getVoicingHintResponse(); },

      /**
       * Set whether or not all responses for this Node will ignore the Properties of responseCollector. If false,
       * all responses will be spoken regardless of responseCollector Properties, which are generally set in user
       * preferences.
       * @public
       */
      setVoicingIgnoreVoicingManagerProperties( ignoreProperties ) {
        this.voicingResponsePacket.ignoreProperties = ignoreProperties;
      },
      set voicingIgnoreVoicingManagerProperties( ignoreProperties ) { this.setVoicingIgnoreVoicingManagerProperties( ignoreProperties ); },

      /**
       * Get whether or not responses are ignoring responseCollector Properties.
       */
      getVoicingIgnoreVoicingManagerProperties() {
        return this.voicingResponsePacket.ignoreProperties;
      },
      get voicingIgnoreVoicingManagerProperties() { return this.getVoicingIgnoreVoicingManagerProperties(); },

      /**
       * Sets the collection of patterns to use for voicing responses, controlling the order, punctuation, and
       * additional content for each combination of response. See ResponsePatternCollection.js if you wish to use
       * a collection of string patterns that are not the default.
       * @public
       *
       * @param {ResponsePatternCollection} patterns - see ResponsePatternCollection
       */
      setVoicingResponsePatternCollection( patterns ) {
        assert && assert( patterns instanceof ResponsePatternCollection );
        this.voicingResponsePacket.responsePatternCollection = patterns;
      },
      set voicingResponsePatternCollection( patterns ) { this.setVoicingResponsePatternCollection( patterns ); },

      /**
       * Get the ResponsePatternCollection object that this Voicing Node is using to collect responses.
       * @public
       *
       * @returns {ResponsePatternCollection}
       */
      getVoicingResponsePatternCollection() {
        return this.voicingResponsePacket.responsePatternCollection;
      },
      get voicingResponsePatternCollection() { return this.getVoicingResponsePatternCollection(); },

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
      getVoicingUtteranceQueue() {
        return this._voicingUtteranceQueue;
      },
      get voicingUtteranceQueue() { return this.getVoicingUtteranceQueue(); },

      /**
       * Called whenever this Node is focused.
       * @public
       *
       * @param {function(SceneryEvent):} focusListener
       */
      setVoicingFocusListener( focusListener ) {
        this._voicingFocusListener = focusListener;
      },

      set voicingFocusListener( utteranceQueue ) { this.setVoicingFocusListener( utteranceQueue ); },

      /**
       * Gets the utteranceQueue through which voicing associated with this Node will be spoken.
       * @public
       *
       * @returns {UtteranceQueue}
       */
      getVoicingFocusListener() {
        return this._voicingFocusListener;
      },
      get voicingFocusListener() { return this.getVoicingFocusListener(); },


      /**
       * The default focus listener attached to this Node during initialization.
       * @public
       */
      defaultFocusListener() {
        this.voicingSpeakFullResponse( {
          contextResponse: null
        } );
      },

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
        this.disposeInteractiveHighlighting();
      }
    } );
  }
};

// @pulic
Voicing.VOICING_OPTION_KEYS = VOICING_OPTION_KEYS;

scenery.register( 'Voicing', Voicing );
export default Voicing;
