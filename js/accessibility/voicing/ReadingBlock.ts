// Copyright 2021-2022, University of Colorado Boulder

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
import Bounds2 from '../../../../dot/js/Bounds2.js';
import Shape from '../../../../kite/js/Shape.js';
import Constructor from '../../../../phet-core/js/types/Constructor.js';
import inheritance from '../../../../phet-core/js/inheritance.js';
import IntentionalAny from '../../../../phet-core/js/IntentionalAny.js';
import responseCollector from '../../../../utterance-queue/js/responseCollector.js';
import ResponsePatternCollection from '../../../../utterance-queue/js/ResponsePatternCollection.js';
import { Focus, Node, ReadingBlockHighlight, ReadingBlockUtterance, scenery, SceneryEvent, Voicing, PDOMInstance, voicingManager, NodeOptions } from '../../imports.js';
import IInputListener from '../../input/IInputListener.js';

const READING_BLOCK_OPTION_KEYS = [
  'readingBlockTagName',
  'readingBlockContent',
  'readingBlockHintResponse',
  'readingBlockActiveHighlight'
];

type ReadingBlockSelfOptions = {
  readingBlockTagName: string | null;
  readingBlockContent: string | null;
  readingBlockHintResponse: string | null;
  readingBlockActiveHighlight: null | Shape | Node;
};

type ReadingBlockOptions = ReadingBlockSelfOptions & NodeOptions;

const CONTENT_HINT_PATTERN = '{{OBJECT}}. {{HINT}}';

/**
 * @param Type
 * @param optionsArgPosition - zero-indexed number that the options argument is provided at
 */
const ReadingBlock = <SuperType extends Constructor>( Type: SuperType, optionsArgPosition: number ) => {

  assert && assert( _.includes( inheritance( Type ), Node ), 'Only Node subtypes should compose Voicing' );

  const ReadingBlockClass = class extends Voicing( Type, optionsArgPosition ) {

    // The tagName used for the ReadingBlock when "Voicing" is enabled, default
    // of button so that it is added to the focus order and can receive 'click' events. You may wish to set this
    // to some other tagName or set to null to remove the ReadingBlock from the focus order. If this is changed,
    // be sure that the ReadingBlock will still respond to `click` events when enabled.
    _readingBlockTagName: string | null;

    // The content for this ReadingBlock that will be spoken by SpeechSynthesis when
    // the ReadingBlock receives input. ReadingBlocks don't use the categories of Voicing content provided by
    // Voicing.ts because ReadingBlocks are always spoken regardless of the Properties of responseCollector.
    _readingBlockContent: string | null;

    // The help content that is read when this ReadingBlock is activated by input,
    // but only when "Helpful Hints" is enabled by the user.
    _readingBlockHintResponse: string | null;

    // The tagName to apply to the Node when voicing is disabled.
    _readingBlockDisabledTagName: string;

    // The highlight that surrounds this ReadingBlock when it is "active" and
    // the Voicing framework is speaking the content associated with this Node. By default, a semi-transparent
    // yellow highlight surrounds this Node's bounds.
    _readingBlockActiveHighlight: null | Shape | Node;

    // (scenery-internal) - Sends a message when the highlight for the ReadingBlock changes. Used
    // by the HighlightOverlay to redraw it if it changes while the highlight is active.
    readingBlockActiveHighlightChangedEmitter: TinyEmitter;

    // Updates the hit bounds of this Node when the local bounds change.
    _localBoundsChangedListener: OmitThisParameter<( localBounds: Bounds2 ) => void>;

    // Triggers activation of the ReadingBlock, requesting speech of its content.
    _readingBlockInputListener: IInputListener;

    // Controls whether the ReadingBlock should be interactive and focusable. At the time of this writing, that is true
    // for all ReadingBlocks when the voicingManager is fully enabled and can speak.
    _readingBlockFocusableChangeListener: OmitThisParameter<( focusable: boolean ) => void>;

    constructor( ...args: IntentionalAny[] ) {

      const providedOptions = ( args[ optionsArgPosition ] || {} ) as ReadingBlockOptions;

      const readingBlockOptions = _.pick( providedOptions, READING_BLOCK_OPTION_KEYS );
      args[ optionsArgPosition ] = _.omit( providedOptions, READING_BLOCK_OPTION_KEYS );

      super( ...args );

      this._readingBlockTagName = 'button';
      this._readingBlockContent = null;
      this._readingBlockHintResponse = null;
      this._readingBlockDisabledTagName = 'p';
      this._readingBlockActiveHighlight = null;
      this.readingBlockActiveHighlightChangedEmitter = new TinyEmitter();

      this._localBoundsChangedListener = this._onLocalBoundsChanged.bind( this );
      ( this as unknown as Node ).localBoundsProperty.link( this._localBoundsChangedListener );

      this._readingBlockInputListener = {
        focus: event => this._speakReadingBlockContent( event ),
        up: event => this._speakReadingBlockContent( event ),
        click: event => this._speakReadingBlockContent( event )
      };

      this._readingBlockFocusableChangeListener = this._onReadingBlockFocusableChanged.bind( this );
      voicingManager.speechAllowedAndFullyEnabledProperty.link( this._readingBlockFocusableChangeListener );

      // All ReadingBlocks have a ReadingBlockHighlight, a focus highlight that is black to indicate it has
      // a different behavior.
      ( this as unknown as Node ).focusHighlight = new ReadingBlockHighlight( this );

      ( this as unknown as Node ).mutate( readingBlockOptions );
    }

    /**
     * Whether a Node composes ReadingBlock.
     */
    get isReadingBlock(): boolean {
      return true;
    }

    /**
     * Set the tagName for the node composing ReadingBlock. This is the tagName (of ParallelDOM) that will be applied
     * to this Node when Reading Blocks are enabled.
     */
    setReadingBlockTagName( tagName: string | null ) {
      this._readingBlockTagName = tagName;
      this._onReadingBlockFocusableChanged( voicingManager.speechAllowedAndFullyEnabledProperty.value );
    }

    set readingBlockTagName( tagName: string | null ) { this.setReadingBlockTagName( tagName ); }

    /**
     * Get the tagName for this Node (of ParallelDOM) when Reading Blocks are enabled.
     */
    getReadingBlockTagName(): string | null {
      return this._readingBlockTagName;
    }

    get readingBlockTagName(): string | null { return this.getReadingBlockTagName(); }

    /**
     * Sets the content that should be read whenever the ReadingBlock receives input that initiates speech.
     */
    setReadingBlockContent( content: string | null ) {
      this._readingBlockContent = content;
    }

    set readingBlockContent( content: string | null ) { this.setReadingBlockContent( content ); }

    /**
     * Gets the content that is spoken whenever the ReadingBLock receives input that would initiate speech.
     */
    getReadingBlockContent(): string | null {
      return this._readingBlockContent;
    }

    get readingBlockContent(): string | null { return this.getReadingBlockContent(); }

    /**
     * Sets the hint response for this ReadingBlock. This is only spoken if "Helpful Hints" are enabled by the user.
     */
    setReadingBlockHintResponse( content: string | null ) {
      this._readingBlockHintResponse = content;
    }

    set readingBlockHintResponse( content: string | null ) { this.setReadingBlockHintResponse( content ); }

    /**
     * Get the hint response for this ReadingBlock. This is additional content that is only read if "Helpful Hints"
     * are enabled.
     */
    getReadingBlockHintResponse(): string | null {
      return this._readingBlockHintResponse;
    }

    get readingBlockHintResponse(): string | null { return this.getReadingBlockHintResponse(); }

    /**
     * Sets the highlight used to surround this Node while the Voicing framework is speaking this content.
     * Do not add this Node to the scene graph, it is added and made visible by the HighlightOverlay.
     */
    setReadingBlockActiveHighlight( readingBlockActiveHighlight: Node | Shape | null ) {
      if ( this._readingBlockActiveHighlight !== readingBlockActiveHighlight ) {
        this._readingBlockActiveHighlight = readingBlockActiveHighlight;

        this.readingBlockActiveHighlightChangedEmitter.emit();
      }
    }

    set readingBlockActiveHighlight( readingBlockActiveHighlight: Node | Shape | null ) { this.setReadingBlockActiveHighlight( readingBlockActiveHighlight ); }

    /**
     * Returns the highlight used to surround this Node when the Voicing framework is reading its
     * content.
     */
    getReadingBlockActiveHighlight(): Node | Shape | null {
      return this._readingBlockActiveHighlight;
    }

    get readingBlockActiveHighlight(): Node | Shape | null { return this._readingBlockActiveHighlight; }

    /**
     * Returns true if this ReadingBlock is "activated", indicating that it has received interaction
     * and the Voicing framework is speaking its content.
     */
    isReadingBlockActivated(): boolean {
      let activated = false;

      const trailIds = Object.keys( this.displays );
      for ( let i = 0; i < trailIds.length; i++ ) {

        const pointerFocus = this.displays[ trailIds[ i ] ].focusManager.readingBlockFocusProperty.value;

        if ( pointerFocus && pointerFocus.trail.lastNode() === this ) {
          activated = true;
          break;
        }
      }
      return activated;
    }

    get readingBlockActivated(): boolean { return this.isReadingBlockActivated(); }

    /**
     * When this Node becomes focusable (because Reading Blocks have just been enabled or disabled), either
     * apply or remove the readingBlockTagName.
     *
     * @param focusable - whether or not ReadingBlocks should be focusable
     */
    _onReadingBlockFocusableChanged( focusable: boolean ) {

      const thisNode = this as unknown as Node;

      thisNode.focusable = focusable;

      if ( focusable ) {
        thisNode.tagName = this._readingBlockTagName;

        // don't add the input listener if we are already active, we may just be updating the tagName in this case
        if ( !thisNode.hasInputListener( this._readingBlockInputListener ) ) {
          thisNode.addInputListener( this._readingBlockInputListener );
        }
      }
      else {
        thisNode.tagName = this._readingBlockDisabledTagName;
        if ( thisNode.hasInputListener( this._readingBlockInputListener ) ) {
          thisNode.removeInputListener( this._readingBlockInputListener );
        }
      }
    }

    /**
     * Update the hit areas for this Node whenever the bounds change.
     */
    _onLocalBoundsChanged( localBounds: Bounds2 ): void {
      const thisNode = this as unknown as Node;
      thisNode.mouseArea = localBounds;
      thisNode.touchArea = localBounds;
    }

    /**
     * Speak the content associated with the ReadingBlock. Sets the readingBlockFocusProperties on
     * the displays so that HighlightOverlays know to activate a highlight while the voicingManager
     * is reading about this Node.
     */
    _speakReadingBlockContent( event: SceneryEvent<Event> ) {

      const displays = ( this as unknown as Node ).getConnectedDisplays();

      const content = this.collectReadingBlockResponses();
      if ( content ) {
        for ( let i = 0; i < displays.length; i++ ) {

          if ( !this.getDescendantsUseHighlighting( event.trail ) ) {

            // the SceneryEvent might have gone through a descendant of this Node
            const rootToSelf = event.trail.subtrailTo( ( this as unknown as Node ) );

            // the trail to a Node may be discontinuous for PDOM events due to pdomOrder,
            // this finds the actual visual trail to use
            const visualTrail = PDOMInstance.guessVisualTrail( rootToSelf, displays[ i ].rootNode );

            const focus = new Focus( displays[ i ], visualTrail );
            const readingBlockUtterance = new ReadingBlockUtterance( focus, {
              alert: content
            } );
            this.speakContent( readingBlockUtterance );
          }
        }
      }
    }

    /**
     * Collect responses for the ReadingBlock, putting together the content and the hint response. The hint response
     * is only read if it exists and hints are enabled by the user. Otherwise, only the readingBlock content will
     * be spoken.
     */
    collectReadingBlockResponses(): string | null {

      const usesHint = this._readingBlockHintResponse && responseCollector.hintResponsesEnabledProperty.value;

      let response = null;
      if ( usesHint ) {

        response = responseCollector.collectResponses( {
          ignoreProperties: true,
          objectResponse: this._readingBlockContent,
          hintResponse: this._readingBlockHintResponse,
          responsePatternCollection: new ResponsePatternCollection( {
            objectHint: CONTENT_HINT_PATTERN
          } )
        } );
      }
      else {
        response = this._readingBlockContent;
      }

      return response;
    }

    dispose() {
      const thisNode = ( this as unknown as Node );
      voicingManager.speechAllowedAndFullyEnabledProperty.unlink( this._readingBlockFocusableChangeListener );
      thisNode.localBoundsProperty.unlink( this._localBoundsChangedListener );

      // remove the input listener that activates the ReadingBlock, only do this if the listener is attached while
      // the ReadingBlock is enabled
      if ( thisNode.hasInputListener( this._readingBlockInputListener ) ) {
        thisNode.removeInputListener( this._readingBlockInputListener );
      }

      super.dispose();
    }
  };


  /**
   * {Array.<string>} - String keys for all of the allowed options that will be set by Node.mutate( options ), in
   * the order they will be evaluated.
   * @protected
   *
   * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
   *       cases that may apply.
   */
  ReadingBlockClass.prototype._mutatorKeys = READING_BLOCK_OPTION_KEYS.concat( ReadingBlockClass.prototype._mutatorKeys );
  assert && assert( ReadingBlockClass.prototype._mutatorKeys.length === _.uniq( ReadingBlockClass.prototype._mutatorKeys ).length,
    'duplicate mutator keys in ReadingBlock' );

  return ReadingBlockClass;
};


scenery.register( 'ReadingBlock', ReadingBlock );
export default ReadingBlock;
