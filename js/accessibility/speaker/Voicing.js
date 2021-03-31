// Copyright 2021, University of Colorado Boulder

/**
 * Trying out a trait for the voicing feature. Thinking about the kinds of options and API this would need to include,
 * most is not implemented yet.
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import inheritance from '../../../../phet-core/js/inheritance.js';
import merge from '../../../../phet-core/js/merge.js';
import Node from '../../nodes/Node.js';
import extend from '../../../../phet-core/js/extend.js';
import scenery from '../../scenery.js';

const Voicing = {
  compose( type ) {
    assert && assert( _.includes( inheritance( type ), Node ), 'Only Node subtypes should compose Voicing' );

    const proto = type.prototype;

    extend( proto, {

      initializeVoicing( options ) {
        options = merge( {

          // {string|null} - The content to be voiced when the Node receives focus. Only spoken when speaking "Object
          // Changes and Screen Text".
          voicingFocusResponse: null,

          // {string|null} - The content to be voiced whenever this Node receives an activation with click, mouse, or
          // touch events.
          voicingActivationResponse: null,

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

          // {Node|Shape|null} - Sets the highlight that will surround this Node when a Pointer is over the
          // voicingHitShape and "interactive highlights" are enabled. A Null value (default) means that we will
          // fall back to use the Node's focus highlight.
          interactiveHighlight: null

          // NOTE: Still need a way to add/remove the Voicing Node from the focus order when voicing is enabled.
          // I think Voicing should handle that for us, optionally. It means that Voicing needs a reference
          // to the preferences Property that controls this. Could pass in that Property directly through options, like
          // focusableProperty: new BooleanProperty( true )
          // then usage would look like
          // focusableProperty: preferencesProperties.voicingEnabledProperty (if thats what controls it)
          // Not too concerned with creating lots of BooleanProperties because not every Node mixes Voicing and we don't
          // and this is still rarely needed for those that do.

        }, options );

        // @public (read-only)
        this.voicing = true;

        this._voicingFocusResponse = options.voicingFocusResponse;
      },

      disposeVoicing() {

      },

      /**
       * @public
       * @param name
       */
      setVoicingFocusResponse( name ) {
        this._voicingFocusResponse = name;
      },

      /**
       * @public
       * @param name
       */
      set voicingFocusResponse( name ) { this.setVoicingFocusResponse( name ); },

      /**
       * @public
       * @returns {Object.voicingFocusResponse|null}
       */
      getVoicingFocusResponse() {
        return this._voicingFocusResponse;
      },

      /**
       * @public
       * @returns {Object.voicingFocusResponse}
       */
      get voicingFocusResponse() { return this.getVoicingFocusResponse(); }
    } );
  }
};

scenery.register( 'Voicing', Voicing );
export default Voicing;
