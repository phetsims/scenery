// Copyright 2021-2022, University of Colorado Boulder

/**
 * A trait that extends Voicing, adding support for "Reading Blocks" of the voicing feature. "Reading Blocks" are
 * UI components in the application that have unique functionality with respect to Voicing.
 *
 *  - Reading Blocks are generally around graphical objects that are not otherwise interactive (like Text).
 *  - They have a unique focus highlight to indicate they can be clicked on to hear voiced content.
 *  - When activated with press or click readingBlockNameResponse is spoken.
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
import { Shape } from '../../../../kite/js/imports.js';
import Constructor from '../../../../phet-core/js/types/Constructor.js';
import inheritance from '../../../../phet-core/js/inheritance.js';
import IntentionalAny from '../../../../phet-core/js/types/IntentionalAny.js';
import ResponsePatternCollection from '../../../../utterance-queue/js/ResponsePatternCollection.js';
import { Focus, Highlight, Node, PDOMInstance, ReadingBlockHighlight, ReadingBlockUtterance, scenery, SceneryEvent, Voicing, voicingManager, VoicingOptions } from '../../imports.js';
import IInputListener from '../../input/IInputListener.js';
import { ResolvedResponse, VoicingResponse } from '../../../../utterance-queue/js/ResponsePacket.js';
import Utterance, { UtteranceOptions } from '../../../../utterance-queue/js/Utterance.js';

const READING_BLOCK_OPTION_KEYS = [
  'readingBlockTagName',
  'readingBlockNameResponse',
  'readingBlockHintResponse',
  'readingBlockResponsePatternCollection',
  'readingBlockActiveHighlight'
];

type SelfOptions = {
  readingBlockTagName?: string | null;
  readingBlockNameResponse?: VoicingResponse;
  readingBlockHintResponse?: VoicingResponse;
  readingBlockResponsePatternCollection?: ResponsePatternCollection;
  readingBlockActiveHighlight?: null | Shape | Node;
};

type UnsupportedVoicingOptions =
  'voicingNameResponse' |
  'voicingObjectResponse' |
  'voicingContextResponse' |
  'voicingHintResponse' |
  'voicingUtterance' |
  'voicingResponsePatternCollection';

export type ReadingBlockOptions = SelfOptions &
  Omit<VoicingOptions, UnsupportedVoicingOptions>;

// Use an assertion signature to narrow the type to ReadingBlockUtterance
function assertReadingBlockUtterance( utterance: Utterance ): asserts utterance is ReadingBlockUtterance {
  if ( !( utterance instanceof ReadingBlockUtterance ) ) {
    assert && assert( false, 'utterance is not a ReadinBlockUtterance' );
  }
}

// An implementation class for ReadingBlock.ts, only used in this class so that we know if we own the Utterance and can
// therefore dispose it.
class OwnedReadingBlockUtterance extends ReadingBlockUtterance {
  constructor( focus: Focus | null, providedOptions?: UtteranceOptions ) {
    super( focus, providedOptions );
  }
}


const DEFAULT_CONTENT_HINT_PATTERN = new ResponsePatternCollection( {
  nameHint: '{{NAME}}. {{HINT}}'
} );

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

    // The tagName to apply to the Node when voicing is disabled.
    _readingBlockDisabledTagName: string;

    // The highlight that surrounds this ReadingBlock when it is "active" and
    // the Voicing framework is speaking the content associated with this Node. By default, a semi-transparent
    // yellow highlight surrounds this Node's bounds.
    _readingBlockActiveHighlight: Highlight;

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
      this._readingBlockDisabledTagName = 'p';
      this._readingBlockActiveHighlight = null;
      this.readingBlockActiveHighlightChangedEmitter = new TinyEmitter();
      this.readingBlockResponsePatternCollection = DEFAULT_CONTENT_HINT_PATTERN;

      this._localBoundsChangedListener = this._onLocalBoundsChanged.bind( this );
      ( this as unknown as Node ).localBoundsProperty.link( this._localBoundsChangedListener );

      this._readingBlockInputListener = {
        focus: event => this._speakReadingBlockContentListener( event ),
        up: event => this._speakReadingBlockContentListener( event ),
        click: event => this._speakReadingBlockContentListener( event )
      };

      this._readingBlockFocusableChangeListener = this._onReadingBlockFocusableChanged.bind( this );
      voicingManager.speechAllowedAndFullyEnabledProperty.link( this._readingBlockFocusableChangeListener );

      // All ReadingBlocks have a ReadingBlockHighlight, a focus highlight that is black to indicate it has
      // a different behavior.
      ( this as unknown as Node ).focusHighlight = new ReadingBlockHighlight( this );

      // All ReadingBlocks use a ReadingBlockUtterance with Focus and Trail data to this Node so that it can be
      // highlighted in the FocusOverlay when this Utterance is being announced.
      this.voicingUtterance = new OwnedReadingBlockUtterance( null );

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
    setReadingBlockTagName( tagName: string | null ): void {
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
    setReadingBlockNameResponse( content: VoicingResponse ) {
      this._voicingResponsePacket.nameResponse = content;
    }

    set readingBlockNameResponse( content: VoicingResponse ) { this.setReadingBlockNameResponse( content ); }

    /**
     * Gets the content that is spoken whenever the ReadingBLock receives input that would initiate speech.
     */
    getReadingBlockNameResponse(): ResolvedResponse {
      return this._voicingResponsePacket.nameResponse;
    }

    get readingBlockNameResponse(): ResolvedResponse { return this.getReadingBlockNameResponse(); }

    /**
     * Sets the hint response for this ReadingBlock. This is only spoken if "Helpful Hints" are enabled by the user.
     */
    setReadingBlockHintResponse( content: VoicingResponse ) {
      this._voicingResponsePacket.hintResponse = content;
    }

    set readingBlockHintResponse( content: VoicingResponse ) { this.setReadingBlockHintResponse( content ); }

    /**
     * Get the hint response for this ReadingBlock. This is additional content that is only read if "Helpful Hints"
     * are enabled.
     */
    getReadingBlockHintResponse(): ResolvedResponse {
      return this._voicingResponsePacket.hintResponse;
    }

    get readingBlockHintResponse(): ResolvedResponse { return this.getReadingBlockHintResponse(); }

    /**
     * Sets the collection of patterns to use for voicing responses, controlling the order, punctuation, and
     * additional content for each combination of response. See ResponsePatternCollection.js if you wish to use
     * a collection of string patterns that are not the default.
     */
    setReadingBlockResponsePatternCollection( patterns: ResponsePatternCollection ) {

      this._voicingResponsePacket.responsePatternCollection = patterns;
    }

    set readingBlockResponsePatternCollection( patterns: ResponsePatternCollection ) { this.setReadingBlockResponsePatternCollection( patterns ); }

    /**
     * Get the ResponsePatternCollection object that this ReadingBlock Node is using to collect responses.
     */
    getReadingBlockResponsePatternCollection(): ResponsePatternCollection {
      return this._voicingResponsePacket.responsePatternCollection;
    }

    get readingBlockResponsePatternCollection(): ResponsePatternCollection { return this.getReadingBlockResponsePatternCollection(); }

    /**
     * ReadingBlock must take a ReadingBlockUtterance for its voicingUtterance. You generally shouldn't be using this.
     * But if you must, you are responsible for setting the ReadingBlockUtterance.readingBlockFocus when this
     * ReadingBlock is activated so that it gets highlighted correctly. See how the default readingBlockFocus is set.
     */
    public override setVoicingUtterance( utterance: ReadingBlockUtterance ) {
      super.setVoicingUtterance( utterance );
    }

    public override set voicingUtterance( utterance: ReadingBlockUtterance ) { super.voicingUtterance = utterance; }

    public override getVoicingUtterance(): ReadingBlockUtterance {
      const utterance = super.getVoicingUtterance();
      assertReadingBlockUtterance( utterance );
      return utterance;
    }

    public override get voicingUtterance(): ReadingBlockUtterance {
      return this.getVoicingUtterance();
    }

    override setVoicingNameResponse(): void { assert && assert( false, 'ReadingBlocks only support setting the name response via readingBlockNameResponse' ); }

    override getVoicingNameResponse(): any { assert && assert( false, 'ReadingBlocks only support getting the name response via readingBlockNameResponse' ); }

    override setVoicingObjectResponse(): void { assert && assert( false, 'ReadingBlocks do not support setting object response' ); }

    override getVoicingObjectResponse(): any { assert && assert( false, 'ReadingBlocks do not support setting object response' ); }

    override setVoicingContextResponse(): void { assert && assert( false, 'ReadingBlocks do not support setting context response' ); }

    override getVoicingContextResponse(): any { assert && assert( false, 'ReadingBlocks do not support setting context response' ); }

    override setVoicingHintResponse(): void { assert && assert( false, 'ReadingBlocks only support setting the hint response via readingBlockHintResponse.' ); }

    override getVoicingHintResponse(): any { assert && assert( false, 'ReadingBlocks only support getting the hint response via readingBlockHintResponse.' ); }

    override setVoicingResponsePatternCollection(): void { assert && assert( false, 'ReadingBlocks only support setting the response patterns via readingBlockResponsePatternCollection.' ); }

    override getVoicingResponsePatternCollection(): any { assert && assert( false, 'ReadingBlocks only support getting the response patterns via readingBlockResponsePatternCollection.' ); }

    /**
     * Sets the highlight used to surround this Node while the Voicing framework is speaking this content.
     * If a Node is provided, do not add this Node to the scene graph, it is added and made visible by the HighlightOverlay.
     */
    setReadingBlockActiveHighlight( readingBlockActiveHighlight: Highlight ) {
      if ( this._readingBlockActiveHighlight !== readingBlockActiveHighlight ) {
        this._readingBlockActiveHighlight = readingBlockActiveHighlight;

        this.readingBlockActiveHighlightChangedEmitter.emit();
      }
    }

    set readingBlockActiveHighlight( readingBlockActiveHighlight: Highlight ) { this.setReadingBlockActiveHighlight( readingBlockActiveHighlight ); }

    /**
     * Returns the highlight used to surround this Node when the Voicing framework is reading its
     * content.
     */
    getReadingBlockActiveHighlight(): Highlight {
      return this._readingBlockActiveHighlight;
    }

    get readingBlockActiveHighlight(): Highlight { return this._readingBlockActiveHighlight; }

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
     * @param focusable - whether ReadingBlocks should be focusable
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
    _speakReadingBlockContentListener( event: SceneryEvent<Event> ) {

      const displays = ( this as unknown as Node ).getConnectedDisplays();

      const readingBlockUtterance = this.voicingUtterance as ReadingBlockUtterance;

      const content = this.collectResponse( {
        nameResponse: this.getReadingBlockNameResponse(),
        hintResponse: this.getReadingBlockHintResponse(),
        ignoreProperties: this.voicingIgnoreVoicingManagerProperties,
        responsePatternCollection: this._voicingResponsePacket.responsePatternCollection,
        utterance: readingBlockUtterance
      } );
      if ( content ) {
        for ( let i = 0; i < displays.length; i++ ) {

          if ( !this.getDescendantsUseHighlighting( event.trail ) ) {

            // the SceneryEvent might have gone through a descendant of this Node
            const rootToSelf = event.trail.subtrailTo( ( this as unknown as Node ) );

            // the trail to a Node may be discontinuous for PDOM events due to pdomOrder,
            // this finds the actual visual trail to use
            const visualTrail = PDOMInstance.guessVisualTrail( rootToSelf, displays[ i ].rootNode );

            const focus = new Focus( displays[ i ], visualTrail );
            readingBlockUtterance.readingBlockFocus = focus;
            this.speakContent( content );
          }
        }
      }
    }

    /**
     * If we created and own the voicingUtterance we can fully dispose of it.
     */
    override _cleanVoicingUtterance() {
      if ( this._voicingUtterance instanceof ReadingBlockUtterance ) {
        this._voicingUtterance.dispose();
      }
      else {
        super._cleanVoicingUtterance();
      }
    }

    override dispose() {
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

// Export a type that lets you check if your Node is composed with ReadingBlock
const wrapper = () => ReadingBlock( Node, 0 );
export type ReadingBlockNode = InstanceType<ReturnType<typeof wrapper>>;

scenery.register( 'ReadingBlock', ReadingBlock );
export default ReadingBlock;
