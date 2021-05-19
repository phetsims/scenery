// Copyright 2021, University of Colorado Boulder

/**
 * A trait for Node that supports the Voicing feature, under accessibility. Allows you to define responses for the Node
 * and make requests to speak that content using HTML5 SpeechSynthesis and the UtteranceQueue. Voicing content is
 * organized into four categories which are responsible for describing different things. Output of this content
 * can be controlled by the voicingManager. These include the
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
 * // REVIEW: I'm not yet sure how Voicing is going to be used just yet, but to me it feels pretty strange that the
 * responses are just strings. I'm trying to think about how we could ever have AccessibleValueHandler.a11yCreateAriaValueText
 * be a string instead of a function that returns a string. Have you thought about expanding the API so that you can provide
 * a function instead of just a string? I worry that if they are just strings, then no one will ever use those values, and
 * will instead just always provide their own strings via override `options` to "collect and speak" functions.
 * https://github.com/phetsims/scenery/issues/1223
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

        // @public (read-only, scenery-internal) - flag indicating that this Node is composed with Voicing functionality
        this.voicing = true;

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
        // by voicingManager Properties. If true, all responses will be spoken when requested, regardless
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
        // with voicingManager. Controls the order of the Voicing responses and even punctuation used when responses
        // are assembled into final content for the UtteranceQueue. See VoicingResponsePatterns for more details.
        this._voicingResponsePatterns = VoicingResponsePatterns.DEFAULT_RESPONSE_PATTERNS;

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
       * speech request. Each response is only spoken if the associated Property of voicingManager is true. If
       * all are Properties are false, nothing will be spoken.
       * @public
       *
       * @param {Object} [options]
       */
      voicingSpeakFullResponse( options ) {

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
       * Each response will only be spoken if the Properties of voicingManager are true. If all of those are false,
       * nothing will be spoken.
       * @public
       *
       * @param {Object} [options]
       */
      voicingSpeakResponse( options ) {

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
       * Speak only the name response assigned to this Node.
       * @param options
       */
      voicingSpeakNameResponse( options ) {

        // options are passed along to collectAndSpeakResponse, see that function for additional options
        options = merge( {
          nameResponse: this._voicingNameResponse
        }, options );

        // REVIEW: Is this dumb to add to each of these methods? What is more buggy, to use voicingSpeakNameResponse() with non-name responses, or to be overly strict for no real reason? https://github.com/phetsims/scenery/issues/1223
        // REVIEW: Another way to handle this is to update the doc to something like, "BY DEFAULT speak only the name response." That way the flexibility seems more like it is built into the function from the beginning.
        // assert && assert( !options.objectResponse, 'only name response should be provided here.' );
        // assert && assert( !options.contextResponse, 'only name response should be provided here.' );
        // assert && assert( !options.hintResponse, 'only name response should be provided here.' );

        this.collectAndSpeakResponse( options );
      },

      /**
       * Speak only the object response assigned to this Node.
       * @public
       *
       * @param {Object} [options]
       */
      voicingSpeakObjectResponse( options ) {

        // options are passed along to collectAndSpeakResponse, see that function for additional options
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

        // options are passed along to collectAndSpeakResponse, see that function for additional options
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

        // options are passed along to collectAndSpeakResponse, see that function for additional options
        options = merge( {
          hintResponse: this._voicingHintResponse
        }, options );

        this.collectAndSpeakResponse( options );
      },

      /**
       * Collect responses with the voicingManager and speak the output with an UtteranceQueue.
       * @protected // REVIEW: Would you expect subtype implementations of the Voicing trait to override this? I didn't see any usages of this, https://github.com/phetsims/scenery/issues/1223
       *
       * @param {Object} [options]
       */
      collectAndSpeakResponse( options ) {
        options = merge( {

          // {boolean} - whether or not this response should ignore the Properties of voicingManager
          ignoreProperties: this._ignoreVoicingManagerProperties, // REVIEW: Wrong var name I believe: https://github.com/phetsims/scenery/issues/1223

          // {Object} - collection of string patterns to use with voicingManager.collectResponses, see
          // VoicingResponsePatterns for more information.
          responsePatterns: this._voicingResponsePatterns,

          // {Utterance|null} - The utterance to use if you want this response to be more controlled in the
          // UtteranceQueue.
          utterance: null
        }, options );

        let response = voicingManager.collectResponses( options );

        // REVIEW: I rewrote this conditional, do you like it? https://github.com/phetsims/scenery/issues/1223
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
       * voicingManager.nameResponsesEnabledProperty is set to true.
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
       * voicingManager.objectResponsesEnabledProperty is set to true.
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
       * @returns {string} // REVIEW: Or null right? https://github.com/phetsims/scenery/issues/1223
       */
      getVoicingObjectResponse() {
        return this._voicingObjectResponse;
      },
      get voicingObjectResponse() { return this.getVoicingObjectResponse(); },

      /**
       * Set the context response for this Node. This is usually the content that describes what has happened in
       * the surrounding application in response to interaction with this Node. When requesting speech, this will
       * only be heard if voicingManager.contextResponsesEnabledProperty is set to true.
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
       * When requesting speech, this will only be spoken when voicingManager.hintResponsesEnabledProperty is set to
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

      // REVIEW: This setter and the getter should be should be voicingUtteranceQueue, as there are potentially multiple utteranceQueues that a single Node could have reference to or use of. https://github.com/phetsims/scenery/issues/1223
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
      }
    } );
  }
};

scenery.register( 'Voicing', Voicing );
export default Voicing;
