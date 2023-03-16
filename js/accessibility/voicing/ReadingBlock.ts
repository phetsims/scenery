// Copyright 2021-2023, University of Colorado Boulder

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
import StrictOmit from '../../../../phet-core/js/types/StrictOmit.js';
import Bounds2 from '../../../../dot/js/Bounds2.js';
import { Shape } from '../../../../kite/js/imports.js';
import Constructor from '../../../../phet-core/js/types/Constructor.js';
import IntentionalAny from '../../../../phet-core/js/types/IntentionalAny.js';
import ResponsePatternCollection from '../../../../utterance-queue/js/ResponsePatternCollection.js';
import { DelayedMutate, Focus, Highlight, Node, PDOMInstance, ReadingBlockHighlight, ReadingBlockUtterance, ReadingBlockUtteranceOptions, scenery, SceneryEvent, Voicing, voicingManager, VoicingOptions } from '../../imports.js';
import TInputListener from '../../input/TInputListener.js';
import { ResolvedResponse, VoicingResponse } from '../../../../utterance-queue/js/ResponsePacket.js';
import Utterance from '../../../../utterance-queue/js/Utterance.js';
import TEmitter from '../../../../axon/js/TEmitter.js';
import memoize from '../../../../phet-core/js/memoize.js';

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
  StrictOmit<VoicingOptions, UnsupportedVoicingOptions>;

// Use an assertion signature to narrow the type to ReadingBlockUtterance
function assertReadingBlockUtterance( utterance: Utterance ): asserts utterance is ReadingBlockUtterance {
  if ( !( utterance instanceof ReadingBlockUtterance ) ) {
    assert && assert( false, 'utterance is not a ReadinBlockUtterance' );
  }
}

// An implementation class for ReadingBlock.ts, only used in this class so that we know if we own the Utterance and can
// therefore dispose it.
class OwnedReadingBlockUtterance extends ReadingBlockUtterance {
  public constructor( focus: Focus | null, providedOptions?: ReadingBlockUtteranceOptions ) {
    super( focus, providedOptions );
  }
}


const DEFAULT_CONTENT_HINT_PATTERN = new ResponsePatternCollection( {
  nameHint: '{{NAME}}. {{HINT}}'
} );

const ReadingBlock = memoize( <SuperType extends Constructor<Node>>( Type: SuperType ) => {

  const ReadingBlockClass = DelayedMutate( 'ReadingBlock', READING_BLOCK_OPTION_KEYS, class ReadingBlockClass extends Voicing( Type ) {

    // The tagName used for the ReadingBlock when "Voicing" is enabled, default
    // of button so that it is added to the focus order and can receive 'click' events. You may wish to set this
    // to some other tagName or set to null to remove the ReadingBlock from the focus order. If this is changed,
    // be sure that the ReadingBlock will still respond to `click` events when enabled.
    private _readingBlockTagName: string | null;

    // The tagName to apply to the Node when voicing is disabled.
    private readonly _readingBlockDisabledTagName: string;

    // The highlight that surrounds this ReadingBlock when it is "active" and
    // the Voicing framework is speaking the content associated with this Node. By default, a semi-transparent
    // yellow highlight surrounds this Node's bounds.
    private _readingBlockActiveHighlight: Highlight;

    // (scenery-internal) - Sends a message when the highlight for the ReadingBlock changes. Used
    // by the HighlightOverlay to redraw it if it changes while the highlight is active.
    public readingBlockActiveHighlightChangedEmitter: TEmitter;

    // Updates the hit bounds of this Node when the local bounds change.
    private readonly _localBoundsChangedListener: OmitThisParameter<( localBounds: Bounds2 ) => void>;

    // Triggers activation of the ReadingBlock, requesting speech of its content.
    private readonly _readingBlockInputListener: TInputListener;

    // Controls whether the ReadingBlock should be interactive and focusable. At the time of this writing, that is true
    // for all ReadingBlocks when the voicingManager is fully enabled and can speak.
    private readonly _readingBlockFocusableChangeListener: OmitThisParameter<( focusable: boolean ) => void>;

    public constructor( ...args: IntentionalAny[] ) {
      super( ...args );

      this._readingBlockTagName = 'button';
      this._readingBlockDisabledTagName = 'p';
      this._readingBlockActiveHighlight = null;
      this.readingBlockActiveHighlightChangedEmitter = new TinyEmitter();
      this.readingBlockResponsePatternCollection = DEFAULT_CONTENT_HINT_PATTERN;

      this._localBoundsChangedListener = this._onLocalBoundsChanged.bind( this );
      this.localBoundsProperty.link( this._localBoundsChangedListener );

      this._readingBlockInputListener = {
        focus: event => this._speakReadingBlockContentListener( event ),
        up: event => this._speakReadingBlockContentListener( event ),
        click: event => this._speakReadingBlockContentListener( event )
      };

      this._readingBlockFocusableChangeListener = this._onReadingBlockFocusableChanged.bind( this );
      voicingManager.speechAllowedAndFullyEnabledProperty.link( this._readingBlockFocusableChangeListener );

      // All ReadingBlocks have a ReadingBlockHighlight, a focus highlight that is black to indicate it has
      // a different behavior.
      this.focusHighlight = new ReadingBlockHighlight( this );

      // All ReadingBlocks use a ReadingBlockUtterance with Focus and Trail data to this Node so that it can be
      // highlighted in the FocusOverlay when this Utterance is being announced.
      this.voicingUtterance = new OwnedReadingBlockUtterance( null );
    }

    /**
     * Whether a Node composes ReadingBlock.
     */
    public get isReadingBlock(): boolean {
      return true;
    }

    /**
     * Set the tagName for the node composing ReadingBlock. This is the tagName (of ParallelDOM) that will be applied
     * to this Node when Reading Blocks are enabled.
     */
    public setReadingBlockTagName( tagName: string | null ): void {
      this._readingBlockTagName = tagName;
      this._onReadingBlockFocusableChanged( voicingManager.speechAllowedAndFullyEnabledProperty.value );
    }

    public set readingBlockTagName( tagName: string | null ) { this.setReadingBlockTagName( tagName ); }

    public get readingBlockTagName(): string | null { return this.getReadingBlockTagName(); }

    /**
     * Get the tagName for this Node (of ParallelDOM) when Reading Blocks are enabled.
     */
    public getReadingBlockTagName(): string | null {
      return this._readingBlockTagName;
    }

    /**
     * Sets the content that should be read whenever the ReadingBlock receives input that initiates speech.
     */
    public setReadingBlockNameResponse( content: VoicingResponse ): void {
      this._voicingResponsePacket.nameResponse = content;
    }

    public set readingBlockNameResponse( content: VoicingResponse ) { this.setReadingBlockNameResponse( content ); }

    public get readingBlockNameResponse(): ResolvedResponse { return this.getReadingBlockNameResponse(); }

    /**
     * Gets the content that is spoken whenever the ReadingBLock receives input that would initiate speech.
     */
    public getReadingBlockNameResponse(): ResolvedResponse {
      return this._voicingResponsePacket.nameResponse;
    }

    /**
     * Sets the hint response for this ReadingBlock. This is only spoken if "Helpful Hints" are enabled by the user.
     */
    public setReadingBlockHintResponse( content: VoicingResponse ): void {
      this._voicingResponsePacket.hintResponse = content;
    }

    public set readingBlockHintResponse( content: VoicingResponse ) { this.setReadingBlockHintResponse( content ); }

    public get readingBlockHintResponse(): ResolvedResponse { return this.getReadingBlockHintResponse(); }

    /**
     * Get the hint response for this ReadingBlock. This is additional content that is only read if "Helpful Hints"
     * are enabled.
     */
    public getReadingBlockHintResponse(): ResolvedResponse {
      return this._voicingResponsePacket.hintResponse;
    }

    /**
     * Sets the collection of patterns to use for voicing responses, controlling the order, punctuation, and
     * additional content for each combination of response. See ResponsePatternCollection.js if you wish to use
     * a collection of string patterns that are not the default.
     */
    public setReadingBlockResponsePatternCollection( patterns: ResponsePatternCollection ): void {

      this._voicingResponsePacket.responsePatternCollection = patterns;
    }

    public set readingBlockResponsePatternCollection( patterns: ResponsePatternCollection ) { this.setReadingBlockResponsePatternCollection( patterns ); }

    public get readingBlockResponsePatternCollection(): ResponsePatternCollection { return this.getReadingBlockResponsePatternCollection(); }

    /**
     * Get the ResponsePatternCollection object that this ReadingBlock Node is using to collect responses.
     */
    public getReadingBlockResponsePatternCollection(): ResponsePatternCollection {
      return this._voicingResponsePacket.responsePatternCollection;
    }

    /**
     * ReadingBlock must take a ReadingBlockUtterance for its voicingUtterance. You generally shouldn't be using this.
     * But if you must, you are responsible for setting the ReadingBlockUtterance.readingBlockFocus when this
     * ReadingBlock is activated so that it gets highlighted correctly. See how the default readingBlockFocus is set.
     */
    public override setVoicingUtterance( utterance: ReadingBlockUtterance ): void {
      super.setVoicingUtterance( utterance );
    }

    public override set voicingUtterance( utterance: ReadingBlockUtterance ) { super.voicingUtterance = utterance; }

    public override get voicingUtterance(): ReadingBlockUtterance { return this.getVoicingUtterance(); }

    public override getVoicingUtterance(): ReadingBlockUtterance {
      const utterance = super.getVoicingUtterance();
      assertReadingBlockUtterance( utterance );
      return utterance;
    }

    public override setVoicingNameResponse(): void { assert && assert( false, 'ReadingBlocks only support setting the name response via readingBlockNameResponse' ); }

    public override getVoicingNameResponse(): IntentionalAny { assert && assert( false, 'ReadingBlocks only support getting the name response via readingBlockNameResponse' ); }

    public override setVoicingObjectResponse(): void { assert && assert( false, 'ReadingBlocks do not support setting object response' ); }

    public override getVoicingObjectResponse(): IntentionalAny { assert && assert( false, 'ReadingBlocks do not support setting object response' ); }

    public override setVoicingContextResponse(): void { assert && assert( false, 'ReadingBlocks do not support setting context response' ); }

    public override getVoicingContextResponse(): IntentionalAny { assert && assert( false, 'ReadingBlocks do not support setting context response' ); }

    public override setVoicingHintResponse(): void { assert && assert( false, 'ReadingBlocks only support setting the hint response via readingBlockHintResponse.' ); }

    public override getVoicingHintResponse(): IntentionalAny { assert && assert( false, 'ReadingBlocks only support getting the hint response via readingBlockHintResponse.' ); }

    public override setVoicingResponsePatternCollection(): void { assert && assert( false, 'ReadingBlocks only support setting the response patterns via readingBlockResponsePatternCollection.' ); }

    public override getVoicingResponsePatternCollection(): IntentionalAny { assert && assert( false, 'ReadingBlocks only support getting the response patterns via readingBlockResponsePatternCollection.' ); }

    /**
     * Sets the highlight used to surround this Node while the Voicing framework is speaking this content.
     * If a Node is provided, do not add this Node to the scene graph, it is added and made visible by the HighlightOverlay.
     */
    public setReadingBlockActiveHighlight( readingBlockActiveHighlight: Highlight ): void {
      if ( this._readingBlockActiveHighlight !== readingBlockActiveHighlight ) {
        this._readingBlockActiveHighlight = readingBlockActiveHighlight;

        this.readingBlockActiveHighlightChangedEmitter.emit();
      }
    }

    public set readingBlockActiveHighlight( readingBlockActiveHighlight: Highlight ) { this.setReadingBlockActiveHighlight( readingBlockActiveHighlight ); }

    public get readingBlockActiveHighlight(): Highlight { return this._readingBlockActiveHighlight; }

    /**
     * Returns the highlight used to surround this Node when the Voicing framework is reading its
     * content.
     */
    public getReadingBlockActiveHighlight(): Highlight {
      return this._readingBlockActiveHighlight;
    }

    /**
     * Returns true if this ReadingBlock is "activated", indicating that it has received interaction
     * and the Voicing framework is speaking its content.
     */
    public isReadingBlockActivated(): boolean {
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

    public get readingBlockActivated(): boolean { return this.isReadingBlockActivated(); }

    /**
     * When this Node becomes focusable (because Reading Blocks have just been enabled or disabled), either
     * apply or remove the readingBlockTagName.
     *
     * @param focusable - whether ReadingBlocks should be focusable
     */
    private _onReadingBlockFocusableChanged( focusable: boolean ): void {
      this.focusable = focusable;

      if ( focusable ) {
        this.tagName = this._readingBlockTagName;

        // don't add the input listener if we are already active, we may just be updating the tagName in this case
        if ( !this.hasInputListener( this._readingBlockInputListener ) ) {
          this.addInputListener( this._readingBlockInputListener );
        }
      }
      else {
        this.tagName = this._readingBlockDisabledTagName;
        if ( this.hasInputListener( this._readingBlockInputListener ) ) {
          this.removeInputListener( this._readingBlockInputListener );
        }
      }
    }

    /**
     * Update the hit areas for this Node whenever the bounds change.
     */
    private _onLocalBoundsChanged( localBounds: Bounds2 ): void {
      this.mouseArea = localBounds;
      this.touchArea = localBounds;
    }

    /**
     * Speak the content associated with the ReadingBlock. Sets the readingBlockFocusProperties on
     * the displays so that HighlightOverlays know to activate a highlight while the voicingManager
     * is reading about this Node.
     */
    private _speakReadingBlockContentListener( event: SceneryEvent ): void {

      const displays = this.getConnectedDisplays();

      const readingBlockUtterance = this.voicingUtterance;

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
            const rootToSelf = event.trail.subtrailTo( this );

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
    protected override cleanVoicingUtterance(): void {
      if ( this._voicingUtterance instanceof OwnedReadingBlockUtterance ) {
        this._voicingUtterance.dispose();
      }
      super.cleanVoicingUtterance();
    }

    public override dispose(): void {
      voicingManager.speechAllowedAndFullyEnabledProperty.unlink( this._readingBlockFocusableChangeListener );
      this.localBoundsProperty.unlink( this._localBoundsChangedListener );

      // remove the input listener that activates the ReadingBlock, only do this if the listener is attached while
      // the ReadingBlock is enabled
      if ( this.hasInputListener( this._readingBlockInputListener ) ) {
        this.removeInputListener( this._readingBlockInputListener );
      }

      super.dispose();
    }

    public override mutate( options?: SelfOptions & Parameters<InstanceType<SuperType>[ 'mutate' ]>[ 0 ] ): this {
      return super.mutate( options );
    }
  } );

  /**
   * {Array.<string>} - String keys for all the allowed options that will be set by Node.mutate( options ), in
   * the order they will be evaluated.
   *
   * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
   *       cases that may apply.
   */
  ReadingBlockClass.prototype._mutatorKeys = READING_BLOCK_OPTION_KEYS.concat( ReadingBlockClass.prototype._mutatorKeys );
  assert && assert( ReadingBlockClass.prototype._mutatorKeys.length === _.uniq( ReadingBlockClass.prototype._mutatorKeys ).length,
    'x mutator keys in ReadingBlock' );

  return ReadingBlockClass;
} );

// Export a type that lets you check if your Node is composed with ReadingBlock
const wrapper = () => ReadingBlock( Node );
export type ReadingBlockNode = InstanceType<ReturnType<typeof wrapper>>;

scenery.register( 'ReadingBlock', ReadingBlock );
export default ReadingBlock;
