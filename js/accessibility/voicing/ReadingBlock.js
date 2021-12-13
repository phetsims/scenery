// Copyright 2021, University of Colorado Boulder

/**
 * A trait that extends Voicing, adding support for "Reading Blocks" of the voicing feature. "Reading Blocks" are
 * UI components in the application that have unique functionality with respect to Voicing.
 *
 *  - Reading Blocks are generally around graphical objects that are not otherwise interactive (like Text).
 *  - They have a unique focus highlight to indicate they can be clicked on to hear voiced content.
 *  - When activated with press or click readingBlockContent is spoken.
 *  - ReadingBlock content is always spoken if the voicingManager is enabled, ignoring Properties of responseCollector.
 *  - While speaking, a yellow highlight will appear over the Node composed with ReadingBlock.
 *  - While voicing is enabled, reading blocks will be added to the focus order.
 *
 * This trait is to be composed with Nodes and assumes that the Node is composed with ParallelDOM.  It uses Node to
 * support mouse/touch input and ParallelDOM to support being added to the focus order and alternative input.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import TinyEmitter from '../../../../axon/js/TinyEmitter.js';
import Shape from '../../../../kite/js/Shape.js';
import extend from '../../../../phet-core/js/extend.js';
import inheritance from '../../../../phet-core/js/inheritance.js';
import StringUtils from '../../../../phetcommon/js/util/StringUtils.js';
import responseCollector from '../../../../utterance-queue/js/responseCollector.js';
import { Focus, Node, ReadingBlockHighlight, ReadingBlockUtterance, scenery, Voicing, voicingManager } from '../../imports.js';

const READING_BLOCK_OPTION_KEYS = [
  'readingBlockTagName',
  'readingBlockContent',
  'readingBlockHintResponse',
  'readingBlockActiveHighlight'
];

const CONTENT_HINT_PATTERN = '{{readingBlockContent}}. {{hintResponse}}';

const ReadingBlock = {

  /**
   * @public
   * @trait {Node}
   * @mixes {Voicing}
   * @param {function(new:Node)} type - The constructor for Node
   */
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

      /**
       * This should be called in the constructor to initialize ReadingBlock. Note, this must be called before
       * options are mutated, so most often you must call options via `mutate()` instead of passing directly to
       * `super()`.
       */
      initializeReadingBlock() {

        // initialize the parent trait
        this.initializeVoicing();

        // @private {boolean} - to make sure that initializeReadingBlock is called before trying to use trait features
        this.readingBlockInitialized = true;

        // @private {string|null} - The tagName used for the ReadingBlock when "Voicing" is enabled, default
        // of button so that it is added to the focus order and can receive 'click' events. You may wish to set this
        // to some other tagName or set to null to remove the ReadingBlock from the focus order. If this is changed,
        // be be sure that the ReadingBlock will still respond to `click` events when enabled.
        this._readingBlockTagName = 'button';

        // @private {string|null} - The content for this ReadingBlock that will be spoken by SpeechSynthesis when
        // the ReadingBlock receives input. ReadingBlocks don't use the categories of Voicing content provided by
        // Voicing.js because ReadingBlocks are always spoken regardless of the Properties of responseCollector.
        this._readingBlockContent = null;

        // @private {string|null} - The help content that is read when this ReadingBlock is activated by input,
        // but only when "Helpful Hints" is enabled by the user.
        this._readingBlockHintResponse = null;

        // @private {string} - The tagName to apply to the Node when voicing is disabled.
        this._readingBlockDisabledTagName = 'p';

        // @private {null|Shape|Node} - The highlight that surrounds this ReadingBlock when it is "active" and
        // the Voicing framework is speaking the content associated with this Node. By default, a semi-transparent
        // yellow highlight surrounds this Node's bounds.
        this._readingBlockActiveHighlight = null;

        // @public (scenery-internal) {TinyEmitter} - Sends a message when the highlight for the ReadingBlock changes. Used
        // by the HighlightOverlay to redraw it if it changes while the highlight is active.
        this.readingBlockActiveHighlightChangedEmitter = new TinyEmitter();

        // @private {function} - Updates the hit bounds of this Node when the local bounds change.
        this.localBoundsChangedListener = this.onLocalBoundsChanged.bind( this );
        this.localBoundsProperty.link( this.localBoundsChangedListener );

        // @private {Object} - Triggers activation of the ReadingBlock, requesting speech of its content.
        this.readingBlockInputListener = {
          focus: event => this.speakReadingBlockContent( event ),
          up: event => this.speakReadingBlockContent( event ),
          click: event => this.speakReadingBlockContent( event )
        };

        // @private - Controls whether or not the ReadingBlock should be interactiveand focusable.
        // At the time of this writing, that is true for all ReadingBlocks when the voicingManager is
        // fully enabled and can speak.
        this.readingBlockFocusableChangeListener = this.onReadingBlockFocusableChanged.bind( this );
        voicingManager.speechAllowedAndFullyEnabledProperty.link( this.readingBlockFocusableChangeListener );

        // All ReadingBlocks have a ReadingBlockHighlight, a focus highlight that is black to indicate it has
        // a different behavior.
        this.focusHighlight = new ReadingBlockHighlight( this );
      },

      /**
       * Whether or not a Node composes ReadingBlock.
       * @returns {boolean}
       */
      get isReadingBlock() {
        return true;
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
        this.onReadingBlockFocusableChanged( voicingManager.speechAllowedAndFullyEnabledProperty.value );
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
       * Sets the hint response for this ReadingBlock. This is only spoken if "Helpful Hints" are enabled by the user.
       * @public
       *
       * @param {string|null} content
       */
      setReadingBlockHintResponse( content ) {
        this._readingBlockHintResponse = content;
      },
      set readingBlockHintResponse( content ) { this.setReadingBlockHintResponse( content ); },

      /**
       * Get the hint response for this ReadingBlock. This is additional content that is only read if "Helpful Hints"
       * are enabled.
       * @public
       *
       * @returns {string|null}
       */
      getReadingBlockHintResponse() {
        return this._readingBlockHintResponse;
      },
      get readingBlockHintResponse() { return this.getReadingBlockHintResponse(); },

      /**
       * Sets the highlight used to surround this Node while the Voicing framework is speaking this content.
       * Do not add this Node to the scene graph, it is added and made visible by the HighlightOverlay.
       * @public
       *
       * @param readingBlockActiveHighlight
       */
      setReadingBlockActiveHighlight( readingBlockActiveHighlight ) {
        assert && assert( readingBlockActiveHighlight === null ||
                          readingBlockActiveHighlight instanceof Node ||
                          readingBlockActiveHighlight instanceof Shape );

        if ( this._readingBlockActiveHighlight !== readingBlockActiveHighlight ) {
          this._readingBlockActiveHighlight = readingBlockActiveHighlight;

          if ( this.readingBlockInitialized ) {
            this.readingBlockActiveHighlightChangedEmitter.emit();
          }
        }
      },
      set readingBlockActiveHighlight( readingBlockActiveHighlight ) { this.setReadingBlockActiveHighlight( readingBlockActiveHighlight ); },

      /**
       * Returns the highlight used to surround this Node when the Voicing framework is reading its
       * content.
       * @returns {null|Shape|Node}
       */
      getReadingBlockActiveHighlight() {
        return this._readingBlockActiveHighlight;
      },
      get readingBlockActiveHighlight() { return this._readingBlockActiveHighlight; },

      /**
       * Returns true if this ReadingBlock is "activated", indicating that it has received interaction
       * and the Voicing framework is speaking its content.
       * @public
       *
       * @returns {boolean}
       */
      isReadingBlockActivated() {
        let activated = false;

        const trailIds = Object.keys( this._displays );
        for ( let i = 0; i < trailIds.length; i++ ) {
          const pointerFocus = this._displays[ trailIds[ i ] ].focusManager.readingBlockFocusProperty.value;
          if ( pointerFocus && pointerFocus.trail.lastNode() === this ) {
            activated = true;
            break;
          }
        }
        return activated;
      },
      get readingBlockActivated() { return this.isReadingBlockActivated(); },

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
        if ( !this.readingBlockInitialized ) {
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
       * the displays so that HighlightOverlays know to activate a highlight while the voicingManager
       * is reading about this Node.
       * @private
       *
       * @param {SceneryEvent} event
       */
      speakReadingBlockContent( event ) {
        assert && assert( this.readingBlockInitialized, 'ReadingBlock must be initialized before speaking' );

        const displays = this.getConnectedDisplays();

        const content = this.collectReadingBlockResponses();
        if ( content ) {
          for ( let i = 0; i < displays.length; i++ ) {
            if ( !this.getDescendantsUseHighlighting( event.trail ) ) {

              // the SceneryEvent might have gone through a descendant of this Node
              const rootToSelf = event.trail.subtrailTo( this );

              // the trail to a Node may be discontinuous for PDOM events due to pdomOrder,
              // this finds the actual visual trail to use
              const visualTrail = scenery.PDOMInstance.guessVisualTrail( rootToSelf, displays[ i ].rootNode );

              const focus = new Focus( displays[ i ], visualTrail );
              const readingBlockUtterance = new ReadingBlockUtterance( focus, {
                alert: content
              } );
              this.speakContent( readingBlockUtterance );
            }
          }
        }
      },

      /**
       * Collect responses for the ReadingBlock, putting together the content and the hint response. The hint response
       * is only read if it exists and hints are enabled by the user. Otherwise, only the readingBlock content will
       * be spoken.
       * @returns {string}
       */
      collectReadingBlockResponses() {
        assert && assert( this.readingBlockInitialized, 'ReadingBlock must be initialized before collecting responses' );

        const usesHelpContent = this._readingBlockHintResponse && responseCollector.hintResponsesEnabledProperty.value;

        let response = null;
        if ( usesHelpContent ) {
          response = StringUtils.fillIn( CONTENT_HINT_PATTERN, {
            readingBlockContent: this._readingBlockContent,
            hintResponse: this._readingBlockHintResponse
          } );
        }
        else {
          response = this._readingBlockContent;
        }

        return response;
      },

      /**
       * @public
       */
      disposeReadingBlock() {
        this.readingBlockInitialized = false;
        voicingManager.speechAllowedAndFullyEnabledProperty.unlink( this.readingBlockFocusableChangeListener );
        this.localBoundsProperty.unlink( this.localBoundsChangedListener );

        // remove the input listener that activates the ReadingBlock, only do this if the listener is attached while
        // the ReadingBlock is enabled
        if ( this.hasInputListener( this.readingBlockInputListener ) ) {
          this.removeInputListener( this.readingBlockInputListener );
        }

        this.disposeVoicing();
      }
    } );
  }
};

scenery.register( 'ReadingBlock', ReadingBlock );
export default ReadingBlock;
