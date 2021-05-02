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
import Node from '../../nodes/Node.js';
import scenery from '../../scenery.js';
import MouseHighlighting from './MouseHighlighting.js';
import voicingManager from './voicingManager.js';

const CREATE_EMPTY_RESPONSE_CONTENT = event => null;

// options that are supported by Voicing.js. Added to mutator keys so that Voicing properties can be set with mutate.
const VOICING_OPTION_KEYS = [
  'voicingCreateObjectResponse',
  'voicingCreateContextResponse',
  'voicingCreateHintResponse',
  'voicingCreateOverrideResponse',
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

        // initialize "super" Trait to support highlights on mouse input
        this.initializeMouseHighlighting();

        // @public (read-only) - flag indicating that this Node is composed with Voicing functionality
        this.voicing = true;

        // @private {function(event: SceneryEvent):string|null} - Create the content for the Node that will be spoken on
        // down, focus, and click events when the user has selected to hear object responses.
        this._voicingCreateObjectResponse = CREATE_EMPTY_RESPONSE_CONTENT;

        // @private {function(event: SceneryEvent):string|null} - Create the content for the Node that will be spoken on
        // down, focus, and click events when the user has selected to hear context responses.
        this._voicingCreateContextResponse = CREATE_EMPTY_RESPONSE_CONTENT;

        // @private {function(event: SceneryEvent):string|null} - Create the content for the Node that will be spoken on
        // down, focus, and click events when the user has selected to hear hints.
        this._voicingCreateHintResponse = CREATE_EMPTY_RESPONSE_CONTENT;

        // @private {function(event: SceneryEvent):string|null} - Create the content for the Node that will be spoken
        // on down, focus, and click events regardless of what output the user has selected as long as voicing is
        // enabled.
        this._voicingCreateOverrideResponse = CREATE_EMPTY_RESPONSE_CONTENT;

        // @private {UtteranceQueue} - The utteranceQueue that content for this VoicingNode will be spoken through.
        // By default, it will go through the Display's voicingUtteranceQueue, but you may need separate
        // UtteranceQueues for different areas of content in your application to manage complex alerts.
        this._voicingUtteranceQueue = null;

        // @public (scenery-internal, read-only) {boolean} - Whether this Node is currently being spoken about from
        // an "activation" like input from the following listener. HighlightOverlay uses this check to make sure that
        // this is the correct Node to highlight for its "speaking" highlight.
        this.speakingFromActivation = false;

        // @private {Object} - Input listener that implements voicing of content on various activation events
        this.speakContentInputListener = {
          down: event => { this.speakVoicingContent( event ); },
          click: event => { this.speakVoicingContent( event ); },
          focus: event => { this.speakVoicingContent( event ); },
          exit: event => { this.speakingFromActivation = false; },
          blur: event => { this.speakingFromActivation = false; }
        };
        this.addInputListener( this.speakContentInputListener );

        if ( options ) {
          this.mutate( options );
        }
      },

      /**
       * Speak the content from the VoicingNod in response to input.
       * @private
       *
       * @param {SceneryEvent} event
       */
      speakVoicingContent( event ) {
        const response = voicingManager.collectResponses( {
          objectResponse: this.voicingCreateObjectResponse( event ),
          interactionHint: this.voicingCreateHintResponse( event ),
          contextResponse: this.voicingCreateContextResponse( event ),
          overrideResponse: this.voicingCreateOverrideResponse( event )
        } );

        // don't send to utteranceQueue if response is empty
        if ( response ) {
          this._displays.forEach( display => {
            const utteranceQueue = this.utteranceQueue || display.voicingUtteranceQueue;
            utteranceQueue.addToBack( response );
          } );

          this.speakingFromActivation = true;
        }
      },

      /**
       * Set the function that will create an object response for the Node, and is spoken if the user has selected hear
       * object responses.
       * @public
       *
       * @param {function(event: SceneryEvent):(string|null)} createResponse
       */
      setVoicingCreateObjectResponse( createResponse ) {
        this._voicingCreateObjectResponse = createResponse;
      },
      set voicingCreateObjectResponse( createResponse ) { this.setVoicingCreateObjectResponse( createResponse ); },

      /**
       * Gets the function that will create an object response for the Node in response to user input, spoken when
       * the user has selected to hear object responses.
       * @public
       *
       * @returns {function(event: SceneryEvent):(string|null)}
       */
      getVoicingCreateObjectResponse() {
        return this._voicingCreateObjectResponse;
      },
      get voicingCreateObjectResponse() { return this.getVoicingCreateObjectResponse(); },

      /**
       * Sets the function that will create an object response for the Node after user input.
       * @public
       *
       * @param {function(event: SceneryEvent):(string|null)} createResponse
       */
      setVoicingCreateContextResponse( createResponse ) {
        this._voicingCreateContextResponse = createResponse;
      },
      set voicingCreateContextResponse( createResponse ) { this.setVoicingCreateContextResponse( createResponse ); },

      /**
       * Get the function that will create a context response for the Node. Content returned by this function will only
       * be spoken if the user has selected to hear context responses.
       * @public
       *
       * @returns {function(event: SceneryEvent):(string|null)}
       */
      getVoicingCreateContextResponse() {
        return this._voicingCreateContextResponse;
      },
      get voicingCreateContextResponse() { return this.getVoicingCreateContextResponse(); },

      /**
       * Set the function that wll create hint response for the Voicing Node. Content returned by the createResponse
       * function will only be spoken if the user has selected to hear hints.
       * @public
       *
       * @param {function(event: SceneryEvent):(string|null)} createResponse
       */
      setVoicingCreateHintResponse( createResponse ) {
        this._voicingCreateHintResponse = createResponse;
      },
      set voicingCreateHintResponse( createResponse ) { this.setVoicingCreateHintResponse( createResponse ); },

      /**
       * Get the function that will create a hint response for the Voicing Node. This content is only spoken
       * when user has selected to hear hints.
       * @public
       *
       * @returns {function(event: SceneryEvent):(string|null)}
       */
      getVoicingCreateHintResponse() {
        return this._voicingCreateHintResponse;
      },
      get voicingCreateHintResponse() { return this.getVoicingCreateHintResponse(); },

      /**
       * Set the function that will create the override response for the VoicingNode. This response will always be
       * spoken, regardless of user selection.
       * @public
       *
       * @param {function(event: SceneryEvent):(string|null)} createResponse
       */
      setVoicingCreateOverrideResponse( createResponse ) {
        this._voicingCreateOverrideResponse = createResponse;
      },
      set voicingCreateOverrideResponse( createResponse ) { this.setVoicingCreateOverrideResponse( createResponse ); },

      /**
       * Get the function that will create the override response for the VoicingNode. This response will always be
       * spoken, regardless of what speech output levels the user has selected.
       * @public
       *
       * @returns {function(event: SceneryEvent):(string|null)}
       */
      getVoicingCreateOverrideResponse() {
        return this._voicingCreateOverrideResponse;
      },
      get voicingCreateOverrideResponse() { return this.getVoicingCreateOverrideResponse(); },

      /**
       * Gets the Property that controls whether this Node is focusable for the purposes of Voicing.
       * @public
       *
       * @param {BooleanProperty} voicingFocusableProperty
       * @returns {null|BooleanProperty}
       */
      getVoicingFocusableProperty( voicingFocusableProperty ) {
        return this._voicingFocusableProperty;
      },
      get voicingFocusableProperty() { return this.getVoicingFocusableProperty(); },

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
       * @public
       */
      disposeVoicing() {
        this.disposeMouseHighlighting();
      }
    } );
  }
};

scenery.register( 'Voicing', Voicing );
export default Voicing;
