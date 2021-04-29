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

const CREATE_EMPTY_RESPONSE_CONTENT = event => null;

// options that are supported by Voicing.js. Added to mutator keys so that Voicing properties can be set with mutate.
const VOICING_OPTION_KEYS = [
  'voicingCreateObjectResponse',
  'voicingCreateContextResponse',
  'voicingCreateHintResponse',
  'voicingCreateOverrideResponse',
  'voicingHighlight',
  'voicingFocusableProperty',
  'voicingTagName',
  'utteranceQueue'
];

const Voicing = {
  compose( type ) {
    assert && assert( _.includes( inheritance( type ), Node ), 'Only Node subtypes should compose Voicing' );

    const proto = type.prototype;

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

        // @public (read-only)
        this.voicing = true;

        // @private {function(event: SceneryEvent):string|null} - Create the content for the Node that will be spoken on
        // down, focus, and click events when the user has selected to hear object responses.
        this._voicingCreateObjectResponse = CREATE_EMPTY_RESPONSE_CONTENT;

        // @private {VoicingHighlight|null} - Sets the highlight that will surround this Node when a Pointer is over the
        // voicingHitShape when voicing is enabled. Typically used with Nodes that are not otherwise interactive
        // but have become clickable for the purposes of Voicing. VoicingHighlight is styled differently from
        // other focus highlights to distinguish this. Null value means that NO voicingHighlight will be used,
        // there are no default voicing highlights.
        this._voicingHighlight = null;

        // @private {null|BooleanProperty} - Controls whether this voicingNode is focusable. Generally useful for Nodes
        // that would not otherwise be focusable when the voicing feature is disabled.
        this._voicingFocusableProperty = null;

        // @private {string|null} - The tagName (of ParallelDOM.js) that will be applied to this Node when this Node is
        // focusable.
        this._voicingTagName = null;

        // @private {string|null} - The tagName to apply to the Node when voicing is disabled, reference stored
        // when the voicingTagName is applied.
        // NOTE: This probably doesn't work very well with more complicated orders of setting tagName and voicingTagName.
        this._voicingDisabledTagName = null;

        // @private {function(event: SceneryEvent):string|null} - Create the content for the Node that will be spoken on
        // down, focus, and click events when the user has selected to hear context responses.
        this._voicingCreateContextResponse = CREATE_EMPTY_RESPONSE_CONTENT;

        // @private {function(event: SceneryEvent):string|null} - Create the content for the Node that will be spoken on
        // down, focus, and click events when the user has selected to hear hints.
        this._voicingCreateHintResponse = CREATE_EMPTY_RESPONSE_CONTENT;

        // @private {function(event: SceneryEvent):string|null} - Create the content for the Node that will be spoken
        // on down, focus, and click events regardless of what ouptut the user has selected as long as voicing is
        // enabled.
        this._voicingCreateOverrideResponse = CREATE_EMPTY_RESPONSE_CONTENT;

        // @private {UtteranceQueue} - The utteranceQueue that content for this VoicingNode will be spoken through.
        // By default, it will go through the Display's VoicingUtteranceQueue, but you may need separate
        // UtteranceQueues for different areas of content in your application to manage complex alerts.
        this._voicingUtteranceQueue = null;

        // @private - reference kept so this listener can be added/removed when the voicingFocusableProperty changes
        this.focusableChangeListener = this.onFocusableChange.bind( this );

        if ( options ) {
          this.mutate( options );
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
       * Sets the highlight that is displayed when this Node has focus or a Pointer is over the voicingHitShape. This
       * will also set the focus highlight
       *
       * @public
       *
       * @param {Node|Shape|null} voicingHighlight
       */
      setVoicingHighlight( voicingHighlight ) {
        this._voicingHighlight = voicingHighlight;
      },
      set voicingHighlight( voicingHighlight ) { this.setVoicingHighlight( voicingHighlight ); },

      /**
       * Gets the highlight that is shown when this Node has focus or a Pointer is over the voicingHitShape.
       * @public
       *
       * @returns {null}
       */
      getVoicingHighlight() {
        return this._voicingHighlight;
      },
      get voicingHighlight() { return this.getVoicingHighlight(); },

      /**
       * Set the Property that will control whether this Node is focusable for the purposes of Voicing. Many Nodes
       * with Voicing are already focusable and so this is not necessary. But others are only focusable when
       * the VoicingFeature is enabled, and this allows you to control that.
       *
       * @public
       * @param {BooleanProperty} voicingFocusableProperty
       */
      setVoicingFocusableProperty( voicingFocusableProperty ) {

        if ( voicingFocusableProperty !== this._voicingFocusableProperty ) {

          // remove previous listener to prevent memory leak
          if ( this._voicingFocusableProperty && this._voicingFocusableProperty.hasListener( this.focusableChangeListener ) ) {
            this._voicingFocusableProperty.unlink( this.focusableChangeListener );
          }

          this._voicingFocusableProperty = voicingFocusableProperty;

          if ( voicingFocusableProperty ) {
            this._voicingFocusableProperty.link( this.focusableChangeListener );
          }
        }
      },
      set voicingFocusableProperty( voicingFocusableProperty ) { this.setVoicingFocusableProperty( voicingFocusableProperty ); },

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
       * Set the tagName for the VoicingNode. By defining a voicingTagName, the tagName (of ParallelDOM.js) will
       * be set whenever the voicingFocusableProperty changes.
       *
       * @param {string} tagName
       */
      setVoicingTagName( tagName ) {
        this._voicingTagName = tagName;

        // update focusability and tagName after setting voicingTagName
        const focusable = this._voicingFocusableProperty ? this._voicingFocusableProperty.value : false;
        this.onFocusableChange( focusable );
      },
      set voicingTagName( tagName ) { this.setVoicingTagName( tagName ); },

      /**
       * Get the tagName that will be set on this Node whenever voicingFocusableProperty is true.
       * @public
       *
       * @returns {null|string}
       */
      getVoicingTagName() {
        return this._voicingTagName;
      },
      get voicingTagName() { return this.getVoicingTagName(); },

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
       * When the voicingFocusableProperty changes, updates ParallelDOM properties that make this Node focusable.
       * @private
       * @param focusable
       */
      onFocusableChange( focusable ) {
        this.focusable = focusable;

        if ( this.voicingTagName !== this.tagName ) {
          if ( focusable ) {
            this._voicingDisabledTagName = this.tagName;
            this.tagName = this._voicingTagName;
          }
          else {

            // possible for onFocusableChange to be called before Voicing has been fully initialized (in which case
            // voicingDisabledTagName will be undefined
            this.tagName = this._voicingDisabledTagName || null;
          }
        }
      },

      /**
       * @public
       */
      disposeVoicing() {
        if ( this._voicingFocusableProperty && this._voicingFocusableProperty.hasListener( this.focusableChangeListener ) ) {
          this._voicingFocusableProperty.unlink( this.focusableChangeListener );
        }
      }
    } );
  }
};

scenery.register( 'Voicing', Voicing );
export default Voicing;
