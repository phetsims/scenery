// Copyright 2021, University of Colorado Boulder

/**
 * A trait that extends Voicing, adding support for "Reading Blocks" of the voicing feature. "Reading Blocks" are
 * UI components in the application that have unique functionality with respect to Voicing.
 *
 *  - Reading Blocks are generally around graphical objects that are not otherwise interactive (like Text).
 *  - They have a unique focus highlight to indicate they can be clicked on to hear voiced content.
 *  - When activated with press or click ReadingBlock content is spoken.
 *  - ReadingBlock content is always spoken if the webSpeaker is enabled, ignoring Properties of voicingManager.
 *  - While speaking, a yellow highlight will appear over the Node composed with ReadingBlock.
 *  - While voicing is enabled, reading blocks will be added to the focus order.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import DerivedProperty from '../../../../axon/js/DerivedProperty.js';
import extend from '../../../../phet-core/js/extend.js';
import inheritance from '../../../../phet-core/js/inheritance.js';
import Node from '../../nodes/Node.js';
import scenery from '../../scenery.js';
import Focus from '../Focus.js';
import Voicing from './Voicing.js';
import voicingManager from './voicingManager.js';
import webSpeaker from './webSpeaker.js';

const READING_BLOCK_OPTION_KEYS = [
  'readingBlockTagName',
  'readingBlockContent'
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

      initializeReadingBlock( options ) {

        // initialize the parent trait
        this.initializeVoicing( options );

        // @public (scenery-internal) - a flag to indicate that this Node has ReadingBlock behavior
        this.readingBlock = true;

        // @private {string|null} - The tagName used for the ReadingBlock when "Voicing" is enabled, default
        // of button so that it is added to the focus order.
        this._readingBlockTagName = 'button';

        // @private {string|null} - The content for this ReadingBlock that will be spoken by SpeechSynthesis when
        // the ReadingBlock receives input. ReadingBlocks don't use the categories of Voicing content provided by
        // Voicing.js because ReadingBlocks are always spoken regardless of the Properties of voicingManager.
        this._readingBlockContent = null;

        // @private {string} - The tagName to apply to the Node when voicing is disabled.
        this._readingBlockDisabledTagName = 'p';

        // @private {function} - Updates the hit bounds of this Node when the local bounds change.
        this.localBoundsChangedListener = this.onLocalBoundsChanged.bind( this );
        this.localBoundsProperty.link( this.localBoundsChangedListener );

        // @private {DerivedProperty.<boolean>} - Controls whether or not the ReadingBlock should be focusable. At
        // the time of this writing, that is true when the webSpeaker is enabled and when Voicing is enabled for the
        // "main content" of the application. See voicingManager for a description of
        // voicingManager.mainWindowVoicingEnabledProperty.
        this.readingBlockFocusableProperty = new DerivedProperty( [
          webSpeaker.enabledProperty,
          voicingManager.mainWindowVoicingEnabledProperty
        ], ( enabled, mainWindowEnabled ) => {
          return enabled && mainWindowEnabled;
        } );

        // @private {Object} - Triggers activation of the ReadingBlock, requesting speech of its content.
        this.readingBlockInputListener = {
          focus: event => this.speakReadingBlockContent( event ),
          down: event => this.speakReadingBlockContent( event ),
          click: event => this.speakReadingBlockContent( event )
        };
        this.addInputListener( this.readingBlockInputListener );

        // @private - reference kept so this listener can be added/removed when the readingBlockFocusable changes
        this.readingBlockFocusableChangeListener = this.onReadingBlockFocusableChanged.bind( this );
        this.readingBlockFocusableProperty.link( this.readingBlockFocusableChangeListener );

        // support passing options through initialize
        if ( options ) {
          this.mutate( _.pick( options, READING_BLOCK_OPTION_KEYS ) );
        }
      },

      /**
       * Set the tagName for the ReadingBlockNode. This is the tagName (of ParallelDOM) that will be applied
       * to this Node when Reading Blocks are enabled.
       * @public
       *
       * @param {string|null} tagName
       */
      setReadingBlockTagName( tagName ) {
        this._readingBlockTagName = tagName;

        // update the tagName for the ReadingBlock if we have been initialized, it is easy to pass options
        // to a supertype before initializeReadingBlock is called
        if ( this.readingBlockFocusableProperty ) {
          this.onReadingBlockFocusableChanged( this.readingBlockFocusableProperty.value );
        }
      },
      set readingBlockTagName( tagName ) { this.setReadingBlockTagName( tagName ); },

      /**
       * Get the tagName for this Node (of ParallelDOM) when Reading Blocks are enabled.
       * @public
       *
       * @returns {string|null}
       */
      getReadingBlockTagName() {
        return this._readingBlockTagName;
      },
      get readingBlockTagName() { return this.getReadingBlockTagName(); },

      /**
       * Sets the content that should be read whenever the ReadingBlock receives input that initiates speech.
       * @public
       *
       * @param {string|null} content
       */
      setReadingBlockContent( content ) {
        this._readingBlockContent = content;
      },
      set readingBlockContent( content ) { this.setReadingBlockContent( content ); },

      /**
       * Gets the content that is spoken whenever the ReadingBLock receives input that would initiate speech.
       * @public
       *
       * @returns {string|null}
       */
      getReadingBlockContent() {
        return this._readingBlockContent;
      },
      get readingBlockContent() { return this.getReadingBlockContent(); },

      /**
       * When this Node becomes focusable (because Reading Blocks have just been enabled or disabled), either
       * apply or remove the readingBlockTagName.
       * @private
       *
       * @param {boolean} focusable - whether or not ReadingBlocks should be focusable
       */
      onReadingBlockFocusableChanged( focusable ) {
        this.focusable = focusable;

        if ( focusable ) {
          this.tagName = this._readingBlockTagName;
        }
        else {

          // possible for onReadingBlockFocusableChanged to be called before Voicing has been fully initialized
          this.tagName = this._readingBlockDisabledTagName;
        }
      },

      /**
       * Update the hit areas for this Node whenever the bounds change.
       * @private
       *
       * @param localBounds
       */
      onLocalBoundsChanged( localBounds ) {
        this.mouseArea = localBounds;
        this.touchArea = localBounds;
      },

      /**
       * Speak the content associated with the ReadingBlock. Sets the readingBlockFocusProperties on
       * the displays so that HighlightOverlays know to activate a highlight while the webSpeaker
       * is reading about this Node.
       * @public
       *
       * @param {SceneryEvent} event
       */
      speakReadingBlockContent( event ) {
        this.speakContent( this._readingBlockContent );

        for ( let i = 0; i < this._displays.length; i++ ) {
          const subtrailToThis = event.trail.subtrailTo( this );
          this._displays[ i ].readingBlockFocusProperty.set( new Focus( this._displays[ i ], subtrailToThis ) );
        }
      },

      /**
       * @public
       */
      disposeReadingBlock() {
        this.readingBlockFocusableProperty.unlink( this.readingBlockFocusableChangeListener );
        this.localBoundsProperty.unlink( this.localBoundsChangedListener );
        this.removeInputListener( this.readingBlockInputListener );
        webSpeaker.endSpeakingEmitter.removeListener( this.endSpeakingListener );
        this.disposeVoicing();
      }
    } );
  }
};

scenery.register( 'ReadingBlock', ReadingBlock );
export default ReadingBlock;
