// Copyright 2021, University of Colorado Boulder

/**
 * A trait that extends Voicing, adding support for "Reading Blocks" of the voicing feature. "Reading Blocks" are
 * UI components in the application that have unique functionality with respect to Voicing.
 *
 *  - Reading Blocks are generally around graphical objects that are not otherwise interactive (like Text).
 *  - They have a unique focus highlight to indicate they can be clicked on to hear voiced content.
 *  - When activated with press or click readingBlockContent is spoken.
 *  - ReadingBlock content is always spoken if the webSpeaker is enabled, ignoring Properties of voicingManager.
 *  - While speaking, a yellow highlight will appear over the Node composed with ReadingBlock.
 *  - While voicing is enabled, reading blocks will be added to the focus order.
 *
 * This trait is to be composed with Nodes and assumes that the Node is composed with ParallelDOM.  It uses Node to
 * support mouse/touch input and ParallelDOM to support being added to the focus order and alternative input.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

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
        this.isReadingBlock = true;

        // @private {string|null} - The tagName used for the ReadingBlock when "Voicing" is enabled, default
        // of button so that it is added to the focus order and can receive 'click' events. You may wish to set this
        // to some other tagName or set to null to remove the ReadingBlock from the focus order. If this is changed,
        // be be sure that the ReadingBlock will still respond to `click` events when enabled.
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

        // @private {Object} - Triggers activation of the ReadingBlock, requesting speech of its content.
        this.readingBlockInputListener = {
          focus: event => this.speakReadingBlockContent( event ),
          down: event => this.speakReadingBlockContent( event ),
          click: event => this.speakReadingBlockContent( event )
        };

        // @private - Controls whether or not the ReadingBlock should be interactive for Voicing and
        // focusable. At the time of this writing, that is true for all ReadingBlocks when the
        // voicingManager indicates that voicing is fully enabled, see the Property in voicingManager
        // for more information.
        this.readingBlockFocusableChangeListener = this.onReadingBlockFocusableChanged.bind( this );
        voicingManager.voicingFullyEnabledProperty.link( this.readingBlockFocusableChangeListener );

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
        this.onReadingBlockFocusableChanged( voicingManager.voicingFullyEnabledProperty.value );
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

        // wait until we have been initialized, it is possible to call setters from mutate before properties of
        // ReadingBlock are defined
        if ( !this.isReadingBlock ) {
          return;
        }

        this.focusable = focusable;

        if ( focusable ) {
          this.tagName = this._readingBlockTagName;

          // don't add the input listener if we are already active, we may just be updating the tagName in this case
          if ( !this.hasInputListener( this.readingBlockInputListener ) ) {
            this.addInputListener( this.readingBlockInputListener );
          }
        }
        else {
          this.tagName = this._readingBlockDisabledTagName;
          if ( this.hasInputListener( this.readingBlockInputListener ) ) {
            this.removeInputListener( this.readingBlockInputListener );

          }
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
       * @private
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
        voicingManager.voicingFullyEnabledProperty.unlink( this.readingBlockFocusableChangeListener );
        this.localBoundsProperty.unlink( this.localBoundsChangedListener );

        // remove the input listener that activates the ReadingBlock, only do this if the listener is attached while
        // the ReadingBlock is enabled
        if ( this.hasInputListener( this.readingBlockInputListener ) ) {
          this.removeInputListener( this.readingBlockInputListener );
        }

        webSpeaker.endSpeakingEmitter.removeListener( this.endSpeakingListener );
        this.disposeVoicing();
      }
    } );
  }
};

scenery.register( 'ReadingBlock', ReadingBlock );
export default ReadingBlock;
