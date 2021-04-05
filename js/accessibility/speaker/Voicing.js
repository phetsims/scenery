// Copyright 2021, University of Colorado Boulder

/**
 * A trait for the Voicing feature that can be composed with a Node. Allows you to specify responses that get spoken
 * with speech synthesis in response to input.
 *
 * A work in progress, don't use yet.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import inheritance from '../../../../phet-core/js/inheritance.js';
import merge from '../../../../phet-core/js/merge.js';
import Node from '../../nodes/Node.js';
import extend from '../../../../phet-core/js/extend.js';
import scenery from '../../scenery.js';

// The collection of Shapes that define the hit areas for Voicing. Used by VoicingInputListener to determine
// when the pointer is over a Node that is composed with Voicing.
const VoicingHitShapes = new Map();

const Voicing = {
  compose( type ) {
    assert && assert( _.includes( inheritance( type ), Node ), 'Only Node subtypes should compose Voicing' );

    const proto = type.prototype;

    extend( proto, {

      /**
       * Initialize in the type being composed with Voicing. Call this in the constructor.
       * @param {Object} [options]
       */
      initializeVoicing( options ) {
        options = merge( {

          // {function(event: SceneryEvent):string|null} - Create the content for the Node that will be spoken on
          // down, focus, and click events when the user has selected to hear object responses.
          voicingCreateObjectResponse: event => null,

          // {function(event: SceneryEvent):string|null} - Create the content for the Node that will be spoken on
          // down, focus, and click events when the user has selected to hear context responses.
          voicingCreateContextResponse: event => null,

          // {function(event: SceneryEvent):string|null} - Create the content for the Node that will be spoken on
          // down, focus, and click events when the user has selected to hear hints.
          voicingCreateHintResponse: event => null,

          // {function(event: SceneryEvent):string|null} - Create the content for the Node that will be spoken on down,
          // focus, and click events regardless of what ouptut the user has selected as long as voicing is enabled.
          voicingCreateOverrideResponse: event => null,

          // {Shape|null} The shape used to determine if a pointer is over this Node for the purposes of voicing and
          // highlights. In the local coordinate frame of the Node. Depending on which features are enabled, a highlight
          // may appear over this Node when a Pointer hits this Shape. Used by SpeakerHighlighter.
          voicingHitShape: null,

          // {VoicingHighlight|null} - Sets the highlight that will surround this Node when a Pointer is over the
          // voicingHitShape when voicing is enabled. Typically used with Nodes that are not otherwise interactive
          // but have become clickable for the purposes of Voicing. VoicingHighlight is styled differently from
          // other focus highlights to distinguish this. Null value means that NO voicingHighlight will be used,
          // there are no default voicing highlights.
          voicingHighlight: null,

          // {boolean}
          voicingHighlightOnly: false,

          // {null|BooleanProperty} - Controls whether this voicingNode is focusable. Generally useful for Nodes
          // that would not otherwise be focusable when the voicing feature is disabled.
          voicingFocusableProperty: null,

          // {string|null} - The tagName (of ParallelDOM.js) that will be applied to this Node when this Node is
          // focusable.
          voicingTagName: null
        }, options );

        // @public (read-only)
        this.voicing = true;

        // @private
        this._voicingHitShape = null;
        this._voicingHighlight = null;
        this._voicingFocusableProperty = null;
        this._voicingTagName = null;
        this._voicingCreateObjectResponse = null;
        this._voicingCreateContextResponse = null;
        this._voicingCreateHintResponse = null;
        this._voicingCreateOverrideResponse = null;

        // @private
        this.focusableChangeListener = this.onFocusableChange.bind( this );

        // NOTE: should be using mutate for this
        this.setVoicingHitShape( options.voicingHitShape );
        this.setVoicingHighlight( options.voicingHighlight );
        this.setVoicingTagName( options.voicingTagName );
        this.setVoicingFocusableProperty( options.voicingFocusableProperty );
        this.setVoicingCreateObjectResponse( options.voicingCreateObjectResponse );
        this.setVoicingCreateContextResponse( options.voicingCreateContextResponse );
        this.setVoicingCreateHintResponse( options.voicingCreateHintResponse );
        this.setVoicingCreateOverrideResponse( options.voicingCreateOverrideResponse );
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
       * Sets the hit shape used to determine if a Pointer is over this Node for the purposes of voicing. If the
       * Pointer is over this shape a voicing highlight may appear over the Node. If a down event occurs within
       * this shape, we may speak the voicingActivationResponse if the feature is enabled.
       *
       * NOTE: Setting a voicingHitShape will make this node pickable and set its mouseArea! This is important because
       * in addition to detecting a hit on the Shape we need to do a hit test to make sure that the Node is hittable
       * (visible, not obscured by other Nodes, doesn't have an ancestor that is not pickable).
       *
       * @public
       *
       * @param {Shape} shape - in the local coordinate frame
       */
      setVoicingHitShape( shape ) {
        if ( shape !== this._voicingHitShape ) {
          this._voicingHitShape = shape;

          VoicingHitShapes.set( this, shape );

          if ( shape ) {
            this.pickable = true;
            this.mouseArea = shape;
          }
        }
      },
      set voicingHitShape( shape ) { this.setVoicingHitShape( shape ); },

      /**
       * Gets the Shape that is used to determine if a Pointer is over this Node for the purposes of Voicing.
       * @public
       *
       * @returns {null|Shape}
       */
      getVoicingHitShape() {
        return this._voicingHitShape;
      },
      get voicingHitShape() { return this.getVoicingHitShape(); },

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
       * When the voicingFocusableProperty changes, updates ParallelDOM properties that make this Node focusable.
       * @private
       * @param focusable
       */
      onFocusableChange( focusable ) {
        this.focusable = focusable;
        if ( this.voicingTagName ) {
          this.tagName = focusable ? this.voicingTagName : null;
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

Voicing.VoicingHitShapes = VoicingHitShapes;

scenery.register( 'Voicing', Voicing );
export default Voicing;
