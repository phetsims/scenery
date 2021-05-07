// Copyright 2021, University of Colorado Boulder

/**
 * A trait that extends Voicing, adding support for "Reading Blocks" of the voicing feature. "Reading Blocks" are
 * UI components in the application that have unique functionality with respect to Voicing.
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

      initializeReadingBlock() {

        // initialize the parent trait
        this.initializeVoicing();

        // @public (scenery-internal) - a flag to indicate that this Node has ReadingBlock behavior
        this.readingBlock = true;

        // @private {string|null} - The tagName used for the ReadingBlock when "Voicing" is enabled, default
        // of button so that it is added to the focus order.
        this._readingBlockTagName = 'button';

        // @private {string|null}
        this._readingBlockContent = null;

        // @private {string|null} - The tagName to apply to the Node when voicing is disabled, reference stored
        // when the readingBlockTagName is applied.
        // NOTE: This wouldn't work very well with more complicated orders of setting tagName and readingBlockTagName.
        this._readingBlockDisabledTagName = null;

        this.localBoundsChangedListener = this.onLocalBoundsChanged.bind( this );
        this.localBoundsProperty.link( this.localBoundsChangedListener );

        // @private {DerivedProperty.<boolean>} - controls whether or not Reading Blocks should be focusable from
        // user settings
        this.readingBlockFocusableProperty = new DerivedProperty( [
          webSpeaker.enabledProperty,
          voicingManager.mainWindowVoicingEnabledProperty ], ( enabled, mainWindowEnabled ) => {
          return enabled && mainWindowEnabled;
        } );

        // by default, ReadingBlocks don't use Voicing "responses" because they should always be readable as long
        // as voicing is enabled. ReadingBlock content should always be spoken in response to these events.
        this.readingBlockInputListener = {
          focus: () => this.speakContent( this._readingBlockContent ),
          down: () => this.speakContent( this._readingBlockContent ),
          click: () => this.speakContent( this._readingBlockContent )
        };
        this.addInputListener( this.readingBlockInputListener );

        // @private - reference kept so this listener can be added/removed when the readingBlockFocusable changes
        this.readingBlockFocusableChangeListener = this.onReadingBlockFocusableChanged.bind( this );
        this.readingBlockFocusableProperty.link( this.readingBlockFocusableChangeListener );
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
       * @param {string|null}
       */
      setReadingBlockContent( content ) {
        this._readingBlockContent = content;
      },
      set readingBlockContent( content ) { this.setReadingBlockContent( content ); },

      /**
       * Gets the content that is spoken whenever the ReadingBLock receives input that would initiate speech.
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
       * @param {boolean} focusable
       */
      onReadingBlockFocusableChanged( focusable ) {
        this.focusable = focusable;

        if ( this.readingBlockTagName !== this.tagName ) {
          if ( focusable ) {
            this._readingBlockDisabledTagName = this.tagName;
            this.tagName = this._readingBlockTagName;
          }
          else {

            // possible for onReadingBlockFocusableChanged to be called before Voicing has been fully initialized
            this.tagName = this._readingBlockDisabledTagName || null;
          }
        }
      },

      /**
       * @private
       * @param localBounds
       */
      onLocalBoundsChanged( localBounds ) {
        this.mouseArea = localBounds;
        this.touchArea = localBounds;
      },

      /**
       * @public
       */
      disposeReadingBlock() {
        this.readingBlockFocusableProperty.unlink( this.readingBlockFocusableChangeListener );
        this.localBoundsProperty.unlink( this.localBoundsChangedListener );
        this.removeInputListener( this.readingBlockInputListener );
        this.disposeVoicing();
      }
    } );
  }
};

scenery.register( 'ReadingBlock', ReadingBlock );
export default ReadingBlock;
