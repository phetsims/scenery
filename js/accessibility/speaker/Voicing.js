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

          // {string|null} - The content to be voiced when the object response is spoken in response to focus.
          voicingObjectFocusResponse: null,

          // {string|null} - The content to be voiced when the object response is spoken in response to a mouse
          // click or keyboard activation with enter/spacebar.
          voicingObjectActivationResponse: null,

          // {string|null - The content to be spoken when the context response is spoken. This is on focus, click, and
          // down on voicing Nodes, if context responses are enabled. Generally, context responses are made in
          // some other Property or other change so it is unlikely that this will be used very often
          voicingContextResponse: null,

          // {string|null} - The content to be spoken whenever the hint should be spoken. This is on focus, click,
          // and down on the voicing Nodes, if hints are enabled.
          voicingHintResponse: null,

          // {string|null} - The content to be voiced whenever this Node receives an activation with click, mouse, or
          // touch events. This will be spoken no matter what speech output level the user has selected, as long
          // as voicing is enabled.
          voicingOverrideResponse: null,

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
        this._voicingObjectFocusResponse = null;
        this._voicingObjectActivationResponse = null;
        this._voicingContextResponse = null;
        this._voicingOverrideResponse = null;
        this._voicingHitShape = null;
        this._voicingHighlight = null;
        this._voicingFocusableProperty = null;
        this._voicingTagName = null;
        this._voicingHintResponse = null;

        // @private
        this.focusableChangeListener = this.onFocusableChange.bind( this );

        // NOTE: should be using mutate for this
        this.setVoicingObjectFocusResponse( options.voicingObjectFocusResponse );
        this.setVoicingObjectActivationResponse( options.voicingObjectActivationResponse );
        this.setVoicingContextResponse( options.voicingContextResponse );
        this.setVoicingHintResponse( options.voicingHintResponse );
        this.setVoicingOverrideResponse( options.voicingActivationResponse );
        this.setVoicingHitShape( options.voicingHitShape );
        this.setVoicingHitShape( options.voicingHitShape );
        this.setVoicingHighlight( options.voicingHighlight );
        this.setVoicingTagName( options.voicingTagName );
        this.setVoicingFocusableProperty( options.voicingFocusableProperty );
      },

      /**
       * Sets the response to be spoken by speech synthesis when this Node receives focus and voicing is enabled.
       * @public
       *
       * @param {string} response
       */
      setVoicingObjectFocusResponse( response ) {
        this._voicingObjectFocusResponse = response;
      },
      set voicingObjectFocusResponse( response ) { this.setVoicingObjectFocusResponse( response ); },

      /**
       * Gets the response that is spoken by speech synthesis when this Node receives focus.
       * @public
       *
       * @returns {string|null}
       */
      getVoicingObjectFocusResponse() {
        return this._voicingObjectFocusResponse;
      },
      get voicingObjectFocusResponse() { return this.getVoicingObjectFocusResponse(); },

      /**
       * Sets the response to be spoken by speech synthesis when this Node receives focus and voicing is enabled.
       * @public
       *
       * @param {string} response
       */
      setVoicingObjectActivationResponse( response ) {
        this._voicingObjectActivationResponse = response;
      },
      set voicingObjectActivationResponse( response ) { this.setVoicingObjectActivationResponse( response ); },

      /**
       * Gets the response that is spoken by speech synthesis when this Node receives focus.
       * @public
       *
       * @returns {string|null}
       */
      getVoicingObjectActivationResponse() {
        return this._voicingObjectActivationResponse;
      },
      get voicingObjectActivationResponse() { return this.getVoicingObjectActivationResponse(); },

      /**
       * Sets the context response for the Voicing Node.
       * @public
       *
       * @param {string} response
       */
      setVoicingContextResponse( response ) {
        this._voicingContextResponse = response;
      },
      set voicingContextResponse( response ) { this.setVoicingContextResponse( response ); },

      /**
       * Gets the response that is spoken by speech synthesis when this Node receives focus.
       * @public
       *
       * @returns {string|null}
       */
      getVoicingContextResponse() {
        return this._voicingContextResponse;
      },
      get voicingContextResponse() { return this.getVoicingContextResponse(); },

      /**
       * Sets the response for this Node when it receives an activation (either click or pointer down).
       * @param response
       */
      setVoicingOverrideResponse( response ) {
        this._voicingOverrideResponse = response;
      },
      set voicingOverrideResponse( response ) { this.setVoicingOverrideResponse( response ); },

      /**
       * Get the response for this when an activation event occurs on this Node.
       * @public
       *
       * @returns {string|null}
       */
      getVoicingOverrideResponse() {
        return this._voicingOverrideResponse;
      },
      get voicingOverrideResponse() { return this.getVoicingOverrideResponse(); },

      /**
       * Set the hint response for this Voicing Node.
       * @param {string} response
       */
      setVoicingHintResponse( response ) {
        this._voicingHintResponse = response;
      },
      set voicingHintResponse( response ) { this.setVoicingHintResponse( response ); },

      /**
       * Get the hint response that will be spoken for this Voicing Node.
       * @returns {null|string}
       */
      getVoicingHintResponse() {
        return this._voicingHintResponse;
      },
      get voicingHintResponse() { return this.getVoicingHintResponse(); },

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
