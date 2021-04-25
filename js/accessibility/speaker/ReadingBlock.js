// Copyright 2021, University of Colorado Boulder

/**
 * A trait that extends Voicing, adding support for "Reading Blocks" of the voicing feature. "Reading Blocks" are
 * UI components in the application that have unique functionality with respect to the Voicing feature.
 *  - Reading Blocks are generally around graphical objects that are not otherwise interactive (like Text).
 *  - They have a unique focus highlight to indicate they can be clicked on to hear voiced content.
 *  - When activated with press or click the web synth will speak about the selected component.
 *  - While speaking, a yellow highlight will appear over the Node composed with ReadingBlock.
 *  - While voicing is enabled, reading blocks will be added to the focus order.
 *
 *  NOTE: This feature is in active development and is not ready for production.
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import DerivedProperty from '../../../../axon/js/DerivedProperty.js';
import extend from '../../../../phet-core/js/extend.js';
import inheritance from '../../../../phet-core/js/inheritance.js';
import scenery from '../../scenery.js';
import Node from '../../nodes/Node.js';
import Voicing from './Voicing.js';
import voicingManager from './voicingManager.js';
import webSpeaker from './webSpeaker.js';

// The collection of Shapes that define the hit areas for Voicing. Used by VoicingInputListener to determine
// when the pointer is over a Node that is composed with Voicing.
const ReadingBlockHitShapes = new Map();

const READING_BLOCK_OPTION_KEYS = [
  'readingBlockHitShape'
];

const ReadingBlock = {
  compose( type ) {
    assert && assert( _.includes( inheritance( type ), Node ) );

    const proto = type.prototype;

    // compose with Voicing
    Voicing.compose( type );

    extend( proto, {

      /**
       * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in
       * the order they will be evaluated.
       * @protected
       *
       * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
       *       cases that may apply.
       */
      _mutatorKeys: READING_BLOCK_OPTION_KEYS.concat( proto._mutatorKeys ),

      initializeReadingBlock() {

        // initialize the parent trait
        this.initializeVoicing();

        // @public (scenery-internal) - a flag to indicate that this Node has ReadingBlock behavior
        this.readingBlock = true;

        // @private {Shape|null} The shape used to determine if a pointer is over this Node for the purposes of voicing and
        // highlights. In the local coordinate frame of the Node. Depending on which features are enabled, a highlight
        // may appear over this Node when a Pointer hits this Shape. Used by SpeakerHighlighter.
        this._readingBlockHitShape = null;

        // setter for the voicingFocusableProperty,
        this.voicingFocusableProperty = new DerivedProperty( [ webSpeaker.enabledProperty, voicingManager.mainWindowVoicingEnabledProperty ], ( enabled, mainWindowEnabled ) => {
          return enabled && mainWindowEnabled;
        } );

        this.voicingTagName = 'button';
      },

      /**
       * Sets the hit shape used to determine if a Pointer is over this Node for the purposes of voicing. If the
       * Pointer is over this shape a voicing highlight may appear over the Node. If a down event occurs within
       * this shape, we may speak the voicingActivationResponse if the feature is enabled.
       *
       * NOTE: Setting a readingBlockHitShape will make this node pickable and set its mouseArea! This is important because
       * in addition to detecting a hit on the Shape we need to do a hit test to make sure that the Node is hittable
       * (visible, not obscured by other Nodes, doesn't have an ancestor that is not pickable).
       *
       * @public
       *
       * @param {Shape} shape - in the local coordinate frame
       */
      setReadingBlockHitShape( shape ) {
        if ( shape !== this._readingBlockHitShape ) {
          this._readingBlockHitShape = shape;

          ReadingBlockHitShapes.set( this, shape );

          if ( shape ) {
            this.pickable = true;
            this.mouseArea = shape;
          }
        }
      },
      set readingBlockHitShape( shape ) { this.setReadingBlockHitShape( shape ); },

      /**
       * Gets the Shape that is used to determine if a Pointer is over this Node for the purposes of Voicing.
       * @public
       *
       * @returns {null|Shape}
       */
      getReadingBlockShape() {
        return this._readingBlockHitShape;
      },
      get readingBlockHitShape() { return this.getReadingBlockShape(); }
    } );
  }
};

ReadingBlock.ReadingBlockHitShapes = ReadingBlockHitShapes;

scenery.register( 'ReadingBlock', ReadingBlock );
export default ReadingBlock;
