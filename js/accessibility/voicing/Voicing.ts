// Copyright 2021-2025, University of Colorado Boulder

/**
 * A trait for Node that supports the Voicing feature, under accessibility. Allows you to define responses for the Node
 * and make requests to speak that content using HTML5 SpeechSynthesis and the UtteranceQueue. Voicing content is
 * organized into four categories which are responsible for describing different things. Responses are stored on the
 * composed type: "ResponsePacket." See that file for details about what responses it stores. Output of this content
 * can be controlled by the responseCollector. Responses are defined as the following. . .
 *
 * - "Name" response: The name of the object that uses Voicing. Similar to the "Accessible Name" in web accessibility.
 * - "Object" response: The state information about the object that uses Voicing.
 * - "Context" response: The contextual changes that result from interaction with the Node that uses Voicing.
 * - "Hint" response: A supporting hint that guides the user toward a desired interaction with this Node.
 *
 * See ResponsePacket, as well as the property and setter documentation for each of these responses for more
 * information.
 *
 * Once this content is set, you can make a request to speak it using an UtteranceQueue with one of the provided
 * functions in this Trait. It is up to you to call one of these functions when you wish for speech to be made. The only
 * exception is on the 'focus' event. Every Node that composes Voicing will speak its responses by when it
 * receives focus.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */

import TinyProperty from '../../../../axon/js/TinyProperty.js';
import inheritance from '../../../../phet-core/js/inheritance.js';
import { combineOptions } from '../../../../phet-core/js/optionize.js';
import Constructor from '../../../../phet-core/js/types/Constructor.js';
import IntentionalAny from '../../../../phet-core/js/types/IntentionalAny.js';
import Tandem from '../../../../tandem/js/Tandem.js';
import responseCollector from '../../../../utterance-queue/js/responseCollector.js';
import ResponsePacket, { ResolvedResponse, SpeakableResolvedOptions, VoicingResponse } from '../../../../utterance-queue/js/ResponsePacket.js';
import ResponsePatternCollection from '../../../../utterance-queue/js/ResponsePatternCollection.js';
import Utterance, { TAlertable, UtteranceOptions } from '../../../../utterance-queue/js/Utterance.js';
import type { ParallelDOMOptions, PDOMValueType } from '../../accessibility/pdom/ParallelDOM.js';
import ParallelDOM from '../../accessibility/pdom/ParallelDOM.js';
import type { InteractiveHighlightingOptions } from '../../accessibility/voicing/InteractiveHighlighting.js';
import InteractiveHighlighting from '../../accessibility/voicing/InteractiveHighlighting.js';
import voicingUtteranceQueue from '../../accessibility/voicing/voicingUtteranceQueue.js';
import Instance from '../../display/Instance.js';
import type { SceneryListenerFunction } from '../../input/TInputListener.js';
import Node from '../../nodes/Node.js';
import scenery from '../../scenery.js';
import DelayedMutate from '../../util/DelayedMutate.js';
import type { TInteractiveHighlighting } from './InteractiveHighlighting.js';
import VoicingActivationResponseListener from './VoicingActivationResponseListener.js';

// Helps enforce that the utterance is defined.
function assertUtterance( utterance: Utterance | null ): asserts utterance is Utterance {
  if ( !( utterance instanceof Utterance ) ) {
    throw new Error( 'utterance is not an Utterance' );
  }
}

// An implementation class for Voicing.ts, only used in this class so that we know if we own the Utterance and can
// therefore dispose it.
class OwnedVoicingUtterance extends Utterance {
  public constructor( providedOptions?: UtteranceOptions ) {
    super( providedOptions );
  }
}

// options that are supported by Voicing.js. Added to mutator keys so that Voicing properties can be set with mutate.
const VOICING_OPTION_KEYS = [
  'voicingNameResponse',
  'voicingObjectResponse',
  'voicingContextResponse',
  'voicingHintResponse',
  'voicingUtterance',
  'voicingResponsePatternCollection',
  'voicingIgnoreVoicingManagerProperties',
  'voicingFocusListener',
  'voicingPressable'
];

type SelfOptions = {

  // see ResponsePacket.nameResponse
  voicingNameResponse?: VoicingResponse;

  // see ResponsePacket.objectResponse
  voicingObjectResponse?: VoicingResponse;

  // see ResponsePacket.contextResponse
  voicingContextResponse?: VoicingResponse;

  // see ResponsePacket.hintResponse
  voicingHintResponse?: VoicingResponse;

  // see ResponsePacket.responsePatternCollection
  voicingResponsePatternCollection?: ResponsePatternCollection;

  // see ResponsePacket.ignoreProperties
  voicingIgnoreVoicingManagerProperties?: boolean;

  // Called when this Node is focused to speak voicing responses on focus. See Voicing.defaultFocusListener for default
  // listener.
  voicingFocusListener?: SceneryListenerFunction<FocusEvent> | null;

  // The utterance to use if you want this response to be more controlled in the UtteranceQueue. This Utterance will be
  // used by all responses spoken by this class. Null to not use an Utterance.
  voicingUtterance?: Utterance | null;

  // If true, a VoicingActivationResponseListener will be added to this Node so that responses are spoken
  // when you press on the Node.
  voicingPressable?: boolean;
};

export type VoicingOptions = SelfOptions & InteractiveHighlightingOptions;

export type SpeakingOptions = {
  utterance?: SelfOptions['voicingUtterance'];
} & SpeakableResolvedOptions;

// Normally our project prefers type aliases to interfaces, but interfaces are necessary for correct usage of "this", see https://github.com/phetsims/tasks/issues/1132
// eslint-disable-next-line @typescript-eslint/consistent-type-definitions
export interface TVoicing<SuperType extends Node = Node> extends TInteractiveHighlighting<SuperType> {
  _voicingResponsePacket: ResponsePacket;

  // @mixin-protected - made public for use in the mixin only
  _voicingUtterance: Utterance | null;

  // @mixin-private - private to this file, but public needed for the interface
  _voicingCanSpeakProperty: TinyProperty<boolean>;

  // @mixin-private - private to this file, but public needed for the interface
  _voicingNameResponseOverride: boolean;
  _voicingHintResponseOverride: boolean;

  // @mixin-private
  _voicingPressable: boolean;

  initialize( ...args: IntentionalAny[] ): this;

  voicingSpeakFullResponse( providedOptions?: SpeakingOptions ): void;

  voicingSpeakResponse( providedOptions?: SpeakingOptions ): void;

  voicingSpeakNameResponse( providedOptions?: SpeakingOptions ): void;

  voicingSpeakObjectResponse( providedOptions?: SpeakingOptions ): void;

  voicingSpeakContextResponse( providedOptions?: SpeakingOptions ): void;

  voicingSpeakHintResponse( providedOptions?: SpeakingOptions ): void;

  // @mixin-protected - made public for use in the mixin only
  collectResponse( providedOptions?: SpeakingOptions ): TAlertable;

  speakContent( content: TAlertable ): void;

  setVoicingNameResponse( response: VoicingResponse ): void;

  voicingNameResponse: ResolvedResponse;

  getVoicingNameResponse(): ResolvedResponse;

  applyDefaultNameResponse( accessibleName: VoicingResponse ): void;

  setVoicingObjectResponse( response: VoicingResponse ): void;

  voicingObjectResponse: ResolvedResponse;

  getVoicingObjectResponse(): ResolvedResponse;

  setVoicingContextResponse( response: VoicingResponse ): void;

  voicingContextResponse: ResolvedResponse;

  getVoicingContextResponse(): ResolvedResponse;

  setVoicingHintResponse( response: VoicingResponse ): void;

  voicingHintResponse: ResolvedResponse;

  getVoicingHintResponse(): ResolvedResponse;

  applyDefaultHintResponse( accessibleHelpText: VoicingResponse ): void;

  setVoicingIgnoreVoicingManagerProperties( ignoreProperties: boolean ): void;

  voicingIgnoreVoicingManagerProperties: boolean;

  getVoicingIgnoreVoicingManagerProperties(): boolean;

  setVoicingResponsePatternCollection( patterns: ResponsePatternCollection ): void;

  voicingResponsePatternCollection: ResponsePatternCollection;

  getVoicingResponsePatternCollection(): ResponsePatternCollection;

  setVoicingUtterance( utterance: Utterance ): void;

  voicingUtterance: Utterance;

  getVoicingUtterance(): Utterance;

  setVoicingFocusListener( focusListener: SceneryListenerFunction<FocusEvent> | null ): void;

  voicingFocusListener: SceneryListenerFunction<FocusEvent> | null;

  getVoicingFocusListener(): SceneryListenerFunction<FocusEvent> | null;

  defaultFocusListener(): void;

  // Prefer exported function isVoicing() for better TypeScript support
  get _isVoicing(): true;

  clean(): void;

  // @mixin-protected - made public for use in the mixin only
  cleanVoicingUtterance(): void;

  // Better options type for the subtype implementation that adds mutator keys
  mutate( options?: SelfOptions & Parameters<SuperType[ 'mutate' ]>[ 0 ] ): this;
}

const Voicing = <SuperType extends Constructor<Node>>( Type: SuperType ): SuperType & Constructor<TVoicing<InstanceType<SuperType>>> => {

  assert && assert( _.includes( inheritance( Type ), Node ), 'Only Node subtypes should compose Voicing' );

  const VoicingClass = DelayedMutate( 'Voicing', VOICING_OPTION_KEYS,
    class VoicingClass extends InteractiveHighlighting( Type ) implements TVoicing<InstanceType<SuperType>> {

      // ResponsePacket that holds all the supported responses to be Voiced
      public _voicingResponsePacket!: ResponsePacket;

      // The utterance that all responses are spoken through.
      // @mixin-protected - made public for use in the mixin only
      public _voicingUtterance: Utterance | null;

      // Called when this node is focused.
      private _voicingFocusListener!: SceneryListenerFunction<FocusEvent> | null;

      // A reference to a listener that is added to this node if it is voicingPressable.
      // This listener speaks the responses when the Voicing Node is clicked with a mouse.
      private _voicingActivationListener!: VoicingActivationResponseListener | null;

      // True when this Voicing Node is voicingPressable. See options.
      public _voicingPressable = false;

      // Indicates whether this Node can speak. A Node can speak if self and all of its ancestors are visible and
      // voicingVisible. This is private because its value depends on the state of the Instance tree. Listening to this
      // to change the scene graph state can be incredibly dangerous and buggy, see https://github.com/phetsims/scenery/issues/1615
      // @mixin-private - private to this file, but public needed for the interface
      public _voicingCanSpeakProperty!: TinyProperty<boolean>;

      // A counter that keeps track of visible and voicingVisible Instances of this Node.
      // As long as this value is greater than zero, this Node can speak. See onInstanceVisibilityChange
      // and onInstanceVoicingVisibilityChange for more implementation details.
      private _voicingCanSpeakCount!: number;

      // Flags that indicate that voicing name or hint response have been set manually. If false,
      // a default accessible name or help text from ParallelDOM may be used by applyDefaultNameResponse or
      // applyDefaultHintResponse. See those functions for more information.
      public _voicingNameResponseOverride = false;
      public _voicingHintResponseOverride = false;

      // Called when `canVoiceEmitter` emits for an Instance.
      private readonly _boundInstanceCanVoiceChangeListener: ( canSpeak: boolean ) => void;

      // Whenever an Instance of this Node is added or removed, add/remove listeners that will update the
      // canSpeakProperty.
      private _boundInstancesChangedListener!: ( instance: Instance, added: boolean ) => void;

      // Input listener that speaks content on focus. This is the only input listener added
      // by Voicing, but it is the one that is consistent for all Voicing nodes. On focus, speak the name, object
      // response, and interaction hint.
      private _speakContentOnFocusListener!: { focus: SceneryListenerFunction<FocusEvent> };

      public constructor( ...args: IntentionalAny[] ) {
        super( ...args );

        // Bind the listeners on construction to be added to observables on initialize and removed on clean/dispose.
        // Instances are updated asynchronously in updateDisplay. The bind creates a new function and we need the
        // reference to persist through the completion of initialize and disposal.
        this._boundInstanceCanVoiceChangeListener = this.onInstanceCanVoiceChange.bind( this );

        this._voicingUtterance = null;

        // We only want to call this method, not any subtype implementation
        VoicingClass.prototype.initialize.call( this );
      }

      // Separate from the constructor to support cases where Voicing is used in Poolable Nodes.
      // ...args: IntentionalAny[] because things like RichTextLink need to provide arguments to initialize, and TS complains
      // otherwise
      public initialize( ...args: IntentionalAny[] ): this {

        // @ts-expect-error
        super.initialize && super.initialize( args );

        this._voicingCanSpeakProperty = new TinyProperty<boolean>( true );
        this._voicingResponsePacket = new ResponsePacket();
        this._voicingFocusListener = this.defaultFocusListener;

        this._voicingActivationListener = null;
        this._voicingPressable = false;

        // Sets the default voicingUtterance and makes this.canSpeakProperty a dependency on its ability to announce.
        this.setVoicingUtterance( new OwnedVoicingUtterance() );

        this._voicingCanSpeakCount = 0;

        this._boundInstancesChangedListener = this.addOrRemoveInstanceListeners.bind( this );

        // This is potentially dangerous to listen to generally, but in this case it is safe because the state we change
        // will only affect how we voice (part of the audio view), and not part of this display's scene graph.
        this.changedInstanceEmitter.addListener( this._boundInstancesChangedListener );

        this._speakContentOnFocusListener = {
          focus: event => {
            this._voicingFocusListener && this._voicingFocusListener( event );
          }
        };
        this.addInputListener( this._speakContentOnFocusListener );

        return this;
      }

      /**
       * Speak all responses assigned to this Node. Options allow you to override a responses for this particular
       * speech request. Each response is only spoken if the associated Property of responseCollector is true. If
       * all are Properties are false, nothing will be spoken.
       */
      public voicingSpeakFullResponse( providedOptions?: SpeakingOptions ): void {

        // options are passed along to collectAndSpeakResponse, see that function for additional options
        const options = combineOptions<SpeakingOptions>( {}, providedOptions );

        // Lazily formulate strings only as needed
        if ( !options.hasOwnProperty( 'nameResponse' ) ) {
          options.nameResponse = this._voicingResponsePacket.nameResponse;
        }
        if ( !options.hasOwnProperty( 'objectResponse' ) ) {
          options.objectResponse = this._voicingResponsePacket.objectResponse;
        }
        if ( !options.hasOwnProperty( 'contextResponse' ) ) {
          options.contextResponse = this._voicingResponsePacket.contextResponse;
        }
        if ( !options.hasOwnProperty( 'hintResponse' ) ) {
          options.hintResponse = this._voicingResponsePacket.hintResponse;
        }

        this.collectAndSpeakResponse( options );
      }

      /**
       * Speak ONLY the provided responses that you pass in with options. This will NOT speak the name, object,
       * context, or hint responses assigned to this node by default. But it allows for clarity at usages so it is
       * clear that you are only requesting certain responses. If you want to speak all of the responses assigned
       * to this Node, use voicingSpeakFullResponse().
       *
       * Each response will only be spoken if the Properties of responseCollector are true. If all of those are false,
       * nothing will be spoken.
       */
      public voicingSpeakResponse( providedOptions?: SpeakingOptions ): void {

        // options are passed along to collectAndSpeakResponse, see that function for additional options
        const options = combineOptions<SpeakingOptions>( {
          nameResponse: null,
          objectResponse: null,
          contextResponse: null,
          hintResponse: null
        }, providedOptions );

        this.collectAndSpeakResponse( options );
      }

      /**
       * By default, speak the name response. But accepts all other responses through options. Respects responseCollector
       * Properties, so the name response may not be spoken if responseCollector.nameResponseEnabledProperty is false.
       */
      public voicingSpeakNameResponse( providedOptions?: SpeakingOptions ): void {

        // options are passed along to collectAndSpeakResponse, see that function for additional options
        const options = combineOptions<SpeakingOptions>( {}, providedOptions );

        // Lazily formulate strings only as needed
        if ( !options.hasOwnProperty( 'nameResponse' ) ) {
          options.nameResponse = this._voicingResponsePacket.nameResponse;
        }

        this.collectAndSpeakResponse( options );
      }

      /**
       * By default, speak the object response. But accepts all other responses through options. Respects responseCollector
       * Properties, so the object response may not be spoken if responseCollector.objectResponseEnabledProperty is false.
       */
      public voicingSpeakObjectResponse( providedOptions?: SpeakingOptions ): void {

        // options are passed along to collectAndSpeakResponse, see that function for additional options
        const options = combineOptions<SpeakingOptions>( {}, providedOptions );

        // Lazily formulate strings only as needed
        if ( !options.hasOwnProperty( 'objectResponse' ) ) {
          options.objectResponse = this._voicingResponsePacket.objectResponse;
        }

        this.collectAndSpeakResponse( options );
      }

      /**
       * By default, speak the context response. But accepts all other responses through options. Respects
       * responseCollector Properties, so the context response may not be spoken if
       * responseCollector.contextResponseEnabledProperty is false.
       */
      public voicingSpeakContextResponse( providedOptions?: SpeakingOptions ): void {

        // options are passed along to collectAndSpeakResponse, see that function for additional options
        const options = combineOptions<SpeakingOptions>( {}, providedOptions );

        // Lazily formulate strings only as needed
        if ( !options.hasOwnProperty( 'contextResponse' ) ) {
          options.contextResponse = this._voicingResponsePacket.contextResponse;
        }

        this.collectAndSpeakResponse( options );
      }

      /**
       * By default, speak the hint response. But accepts all other responses through options. Respects
       * responseCollector Properties, so the hint response may not be spoken if
       * responseCollector.hintResponseEnabledProperty is false.
       */
      public voicingSpeakHintResponse( providedOptions?: SpeakingOptions ): void {

        // options are passed along to collectAndSpeakResponse, see that function for additional options
        const options = combineOptions<SpeakingOptions>( {}, providedOptions );

        // Lazily formulate strings only as needed
        if ( !options.hasOwnProperty( 'hintResponse' ) ) {
          options.hintResponse = this._voicingResponsePacket.hintResponse;
        }

        this.collectAndSpeakResponse( options );
      }

      /**
       * Collect responses with the responseCollector and speak the output with an UtteranceQueue.
       */
      private collectAndSpeakResponse( providedOptions?: SpeakingOptions ): void {
        this.speakContent( this.collectResponse( providedOptions ) );
      }

      /**
       * Combine all types of response into a single alertable, potentially depending on the current state of
       * responseCollector Properties (filtering what kind of responses to present in the resolved response).
       * @mixin-protected - made public for use in the mixin only
       */
      public collectResponse( providedOptions?: SpeakingOptions ): TAlertable {
        const options = combineOptions<SpeakingOptions>( {
          ignoreProperties: this._voicingResponsePacket.ignoreProperties,
          responsePatternCollection: this._voicingResponsePacket.responsePatternCollection,
          utterance: this.voicingUtterance
        }, providedOptions );

        let response: TAlertable = responseCollector.collectResponses( options );

        if ( options.utterance ) {
          options.utterance.alert = response;
          response = options.utterance;
        }
        return response;
      }

      /**
       * Use the provided function to create content to speak in response to input. The content is then added to the
       * back of the voicing UtteranceQueue.
       */
      public speakContent( content: TAlertable ): void {

        const notPhetioArchetype = !Tandem.PHET_IO_ENABLED || !this.isInsidePhetioArchetype();

        // don't send to utteranceQueue if response is empty
        // don't send to utteranceQueue for PhET-iO dynamic element archetypes, https://github.com/phetsims/joist/issues/817
        if ( content && notPhetioArchetype ) {
          voicingUtteranceQueue.addToBack( content );
        }
      }

      /**
       * Sets the voicingNameResponse for this Node. This is usually the label of the element and is spoken
       * when the object receives input. When requesting speech, this will only be spoken if
       * responseCollector.nameResponsesEnabledProperty is set to true.
       */
      public setVoicingNameResponse( response: VoicingResponse ): void {
        this._voicingNameResponseOverride = response !== null;
        this._voicingResponsePacket.nameResponse = response;
      }

      public set voicingNameResponse( response: VoicingResponse ) { this.setVoicingNameResponse( response ); }

      public get voicingNameResponse(): ResolvedResponse { return this.getVoicingNameResponse(); }

      /**
       * Get the voicingNameResponse for this Node.
       */
      public getVoicingNameResponse(): ResolvedResponse {
        return this._voicingResponsePacket.nameResponse;
      }

      /**
       * Sets the name response for this VoicingNode to match the provided accessibleName. If setAccessibleName
       * has been used once, this is a no-op.
       *
       * This should rarely be used. It is intended for use cases where the name response should match the
       * ParallelDOM.accessibleName to align the APIs. See Voicing.BASIC_ACCESSIBLE_NAME_BEHAVIOR. It is probably most
       * useful in your own behavior function.
       */
      public applyDefaultNameResponse( accessibleName: VoicingResponse ): void {

        // If the voicingNameResponse has not been set manually, it will take the same value
        // as the accessibleName.
        if ( !this._voicingNameResponseOverride ) {
          this._voicingResponsePacket.nameResponse = accessibleName;
        }
      }

      /**
       * Set the object response for this Node. This is usually the state information associated with this Node, such
       * as its current input value. When requesting speech, this will only be heard when
       * responseCollector.objectResponsesEnabledProperty is set to true.
       */
      public setVoicingObjectResponse( response: VoicingResponse ): void {
        this._voicingResponsePacket.objectResponse = response;
      }

      public set voicingObjectResponse( response: VoicingResponse ) { this.setVoicingObjectResponse( response ); }

      public get voicingObjectResponse(): ResolvedResponse { return this.getVoicingObjectResponse(); }

      /**
       * Gets the object response for this Node.
       */
      public getVoicingObjectResponse(): ResolvedResponse {
        return this._voicingResponsePacket.objectResponse;
      }

      /**
       * Set the context response for this Node. This is usually the content that describes what has happened in
       * the surrounding application in response to interaction with this Node. When requesting speech, this will
       * only be heard if responseCollector.contextResponsesEnabledProperty is set to true.
       */
      public setVoicingContextResponse( response: VoicingResponse ): void {
        this._voicingResponsePacket.contextResponse = response;
      }

      public set voicingContextResponse( response: VoicingResponse ) { this.setVoicingContextResponse( response ); }

      public get voicingContextResponse(): ResolvedResponse { return this.getVoicingContextResponse(); }

      /**
       * Gets the context response for this Node.
       */
      public getVoicingContextResponse(): ResolvedResponse {
        return this._voicingResponsePacket.contextResponse;
      }

      /**
       * Sets the hint response for this Node. This is usually a response that describes how to interact with this Node.
       * When requesting speech, this will only be spoken when responseCollector.hintResponsesEnabledProperty is set to
       * true.
       */
      public setVoicingHintResponse( response: VoicingResponse ): void {
        this._voicingHintResponseOverride = response !== null;
        this._voicingResponsePacket.hintResponse = response;
      }

      public set voicingHintResponse( response: VoicingResponse ) { this.setVoicingHintResponse( response ); }

      public get voicingHintResponse(): ResolvedResponse { return this.getVoicingHintResponse(); }

      /**
       * Gets the hint response for this Node.
       */
      public getVoicingHintResponse(): ResolvedResponse {
        return this._voicingResponsePacket.hintResponse;
      }

      /**
       * Sets the hint response for this VoicingNode to match the provided help text. If setVoicingHintResponse
       * has been used once, this is a no-op.
       *
       * This should rarely be used. It is intended for use cases where the hint response should match the
       * ParallelDOM.accessibleHelpText to align the APIs. See Voicing.BASIC_HELP_TEXT_BEHAVIOR. It is probably most useful in
       * your own behavior function.
       */
      public applyDefaultHintResponse( accessibleHelpText: VoicingResponse ): void {

        // If the voicingHintResponse has not been set manually, it will take the same value
        // as the accessibleHelpText.
        if ( !this._voicingHintResponseOverride ) {
          this._voicingResponsePacket.hintResponse = accessibleHelpText;
        }
      }

      /**
       * Set whether or not all responses for this Node will ignore the Properties of responseCollector. If false,
       * all responses will be spoken regardless of responseCollector Properties, which are generally set in user
       * preferences.
       */
      public setVoicingIgnoreVoicingManagerProperties( ignoreProperties: boolean ): void {
        this._voicingResponsePacket.ignoreProperties = ignoreProperties;
      }

      public set voicingIgnoreVoicingManagerProperties( ignoreProperties: boolean ) { this.setVoicingIgnoreVoicingManagerProperties( ignoreProperties ); }

      public get voicingIgnoreVoicingManagerProperties(): boolean { return this.getVoicingIgnoreVoicingManagerProperties(); }

      /**
       * Get whether or not responses are ignoring responseCollector Properties.
       */
      public getVoicingIgnoreVoicingManagerProperties(): boolean {
        return this._voicingResponsePacket.ignoreProperties;
      }

      /**
       * Sets the collection of patterns to use for voicing responses, controlling the order, punctuation, and
       * additional content for each combination of response. See ResponsePatternCollection.js if you wish to use
       * a collection of string patterns that are not the default.
       */
      public setVoicingResponsePatternCollection( patterns: ResponsePatternCollection ): void {

        this._voicingResponsePacket.responsePatternCollection = patterns;
      }

      public set voicingResponsePatternCollection( patterns: ResponsePatternCollection ) { this.setVoicingResponsePatternCollection( patterns ); }

      public get voicingResponsePatternCollection(): ResponsePatternCollection { return this.getVoicingResponsePatternCollection(); }

      /**
       * Get the ResponsePatternCollection object that this Voicing Node is using to collect responses.
       */
      public getVoicingResponsePatternCollection(): ResponsePatternCollection {
        return this._voicingResponsePacket.responsePatternCollection;
      }

      /**
       * Sets the utterance through which voicing associated with this Node will be spoken. By default on initialize,
       * one will be created, but a custom one can optionally be provided.
       */
      public setVoicingUtterance( utterance: Utterance ): void {
        if ( this._voicingUtterance !== utterance ) {

          if ( this._voicingUtterance ) {
            this.cleanVoicingUtterance();
          }

          Voicing.registerUtteranceToVoicingNode( utterance, this );
          this._voicingUtterance = utterance;
        }
      }

      public set voicingUtterance( utterance: Utterance ) { this.setVoicingUtterance( utterance ); }

      public get voicingUtterance(): Utterance { return this.getVoicingUtterance(); }

      /**
       * Gets the utterance through which voicing associated with this Node will be spoken.
       */
      public getVoicingUtterance(): Utterance {
        assertUtterance( this._voicingUtterance );
        return this._voicingUtterance;
      }

      /**
       * When voicingPressable is true, a listener is added to this Node to speak responses when it is
       * pressed with a pointer.
       */
      public setVoicingPressable( pressable: boolean ): void {
        if ( pressable !== this._voicingPressable ) {
          if ( pressable ) {
            this._voicingActivationListener = new VoicingActivationResponseListener( this );
            this.addInputListener( this._voicingActivationListener );
          }
          else if ( !pressable && this._voicingActivationListener ) {
            this.removeInputListener( this._voicingActivationListener );
            this._voicingActivationListener.dispose();
            this._voicingActivationListener = null;
          }

          this._voicingPressable = pressable;
        }
      }

      public set voicingPressable( pressable: boolean ) { this.setVoicingPressable( pressable ); }

      public get voicingPressable(): boolean { return this.getVoicingPressable(); }

      public getVoicingPressable(): boolean {
        return this._voicingPressable;
      }


      /**
       * Called whenever this Node is focused.
       */
      public setVoicingFocusListener( focusListener: SceneryListenerFunction<FocusEvent> | null ): void {
        this._voicingFocusListener = focusListener;
      }

      public set voicingFocusListener( focusListener: SceneryListenerFunction<FocusEvent> | null ) { this.setVoicingFocusListener( focusListener ); }

      public get voicingFocusListener(): SceneryListenerFunction<FocusEvent> | null { return this.getVoicingFocusListener(); }

      /**
       * Gets the utteranceQueue through which voicing associated with this Node will be spoken.
       */
      public getVoicingFocusListener(): SceneryListenerFunction<FocusEvent> | null {
        return this._voicingFocusListener;
      }

      /**
       * The default focus listener attached to this Node during initialization.
       */
      public defaultFocusListener(): void {
        this.voicingSpeakFullResponse( {
          contextResponse: null
        } );
      }

      /**
       * Whether a Node composes Voicing.
       */
      public get _isVoicing(): true {
        return true;
      }

      /**
       * Detaches references that ensure this components of this Trait are eligible for garbage collection.
       */
      public override dispose(): void {
        this.removeInputListener( this._speakContentOnFocusListener );
        this.changedInstanceEmitter.removeListener( this._boundInstancesChangedListener );

        if ( this._voicingPressable ) {
          assert && assert( this._voicingActivationListener, 'voicingActivationListener should be set if voicingPressable is true' );
          const voicingActivationListener = this._voicingActivationListener!;
          this.removeInputListener( voicingActivationListener );
          voicingActivationListener.dispose();
          this._voicingActivationListener = null;
          this._voicingPressable = false;
        }

        if ( this._voicingUtterance ) {
          this.cleanVoicingUtterance();
          this._voicingUtterance = null;
        }

        super.dispose();
      }

      public clean(): void {
        this.removeInputListener( this._speakContentOnFocusListener );
        this.changedInstanceEmitter.removeListener( this._boundInstancesChangedListener );

        if ( this._voicingPressable ) {
          assert && assert( this._voicingActivationListener, 'voicingActivationListener should be set if voicingPressable is true' );
          const voicingActivationListener = this._voicingActivationListener!;
          this.removeInputListener( voicingActivationListener );
          voicingActivationListener.dispose();
          this._voicingActivationListener = null;
          this._voicingPressable = false;
        }

        if ( this._voicingUtterance ) {
          this.cleanVoicingUtterance();
          this._voicingUtterance = null;
        }

        // @ts-expect-error
        super.clean && super.clean();
      }

      /***********************************************************************************************************/
      // PRIVATE METHODS
      /***********************************************************************************************************/

      /**
       * When visibility and voicingVisibility change such that the Instance can now speak, update the counting
       * variable that tracks how many Instances of this VoicingNode can speak. To speak the Instance must be globally\
       * visible and voicingVisible.
       */
      private onInstanceCanVoiceChange( canSpeak: boolean ): void {

        if ( canSpeak ) {
          this._voicingCanSpeakCount++;
        }
        else {
          this._voicingCanSpeakCount--;
        }

        assert && assert( this._voicingCanSpeakCount >= 0, 'the voicingCanSpeakCount should not go below zero' );
        assert && assert( this._voicingCanSpeakCount <= this.instances.length,
          'The voicingCanSpeakCount cannot be greater than the number of Instances.' );

        this._voicingCanSpeakProperty.value = this._voicingCanSpeakCount > 0;
      }

      /**
       * Update the canSpeakProperty and counting variable in response to an Instance of this Node being added or
       * removed.
       */
      private handleInstancesChanged( instance: Instance, added: boolean ): void {
        const isVisible = instance.visible && instance.voicingVisible;
        if ( isVisible ) {

          // If the added Instance was visible and voicingVisible it should increment the counter. If the removed
          // instance is NOT visible/voicingVisible it would not have contributed to the counter so we should not
          // decrement in that case.
          this._voicingCanSpeakCount = added ? this._voicingCanSpeakCount + 1 : this._voicingCanSpeakCount - 1;
        }

        this._voicingCanSpeakProperty.value = this._voicingCanSpeakCount > 0;
      }

      /**
       * Add or remove listeners on an Instance watching for changes to visible or voicingVisible that will modify
       * the voicingCanSpeakCount. See documentation for voicingCanSpeakCount for details about how this controls the
       * voicingCanSpeakProperty.
       */
      private addOrRemoveInstanceListeners( instance: Instance, added: boolean ): void {
        assert && assert( instance.canVoiceEmitter, 'Instance must be initialized.' );

        if ( added ) {
          // @ts-expect-error - Emitters in Instance need typing
          instance.canVoiceEmitter!.addListener( this._boundInstanceCanVoiceChangeListener );
        }
        else {
          // @ts-expect-error - Emitters in Instance need typing
          instance.canVoiceEmitter!.removeListener( this._boundInstanceCanVoiceChangeListener );
        }

        // eagerly update the canSpeakProperty and counting variables in addition to adding change listeners
        this.handleInstancesChanged( instance, added );
      }

      /**
       * Clean this._voicingUtterance, disposing if we own it or unregistering it if we do not.
       * @mixin-protected - made public for use in the mixin only
       */
      public cleanVoicingUtterance(): void {
        assert && assert( this._voicingUtterance, 'A voicingUtterance must be available to clean.' );
        if ( this._voicingUtterance instanceof OwnedVoicingUtterance ) {
          this._voicingUtterance.dispose();
        }
        else if ( this._voicingUtterance && !this._voicingUtterance.isDisposed ) {
          Voicing.unregisterUtteranceToVoicingNode( this._voicingUtterance, this );
        }
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
  VoicingClass.prototype._mutatorKeys = VOICING_OPTION_KEYS.concat( VoicingClass.prototype._mutatorKeys );

  assert && assert( VoicingClass.prototype._mutatorKeys.length === _.uniq( VoicingClass.prototype._mutatorKeys ).length, 'duplicate mutator keys in Voicing' );

  return VoicingClass;
};

Voicing.VOICING_OPTION_KEYS = VOICING_OPTION_KEYS;

/**
 * Alert an Utterance to the voicingUtteranceQueue. The Utterance must have voicingCanAnnounceProperties and hopefully
 * at least one of the Properties is a VoicingNode's canAnnounceProperty so that this Utterance is only announced
 * when the VoicingNode is globally visible and voicingVisible.
 * @static
 */
Voicing.alertUtterance = ( utterance: Utterance ) => {
  assert && assert( utterance.voicingCanAnnounceProperties.length > 0, 'voicingCanAnnounceProperties required, this Utterance might not be connected to Node in the scene graph.' );
  voicingUtteranceQueue.addToBack( utterance );
};

/**
 * Assign the voicingNode's voicingCanSpeakProperty to the Utterance so that the Utterance can only be announced
 * if the voicingNode is globally visible and voicingVisible in the display.
 * @static
 */
Voicing.registerUtteranceToVoicingNode = ( utterance: Utterance, voicingNode: TVoicing ) => {
  const existingCanAnnounceProperties = utterance.voicingCanAnnounceProperties;

  const voicingCanSpeakProperty = voicingNode._voicingCanSpeakProperty;
  if ( !existingCanAnnounceProperties.includes( voicingCanSpeakProperty ) ) {
    utterance.voicingCanAnnounceProperties = existingCanAnnounceProperties.concat( [ voicingCanSpeakProperty ] );
  }
};

/**
 * Remove a voicingNode's voicingCanSpeakProperty from the Utterance.
 * @static
 */
Voicing.unregisterUtteranceToVoicingNode = ( utterance: Utterance, voicingNode: VoicingNode ) => {
  const existingCanAnnounceProperties = utterance.voicingCanAnnounceProperties;

  const voicingCanSpeakProperty = voicingNode._voicingCanSpeakProperty;
  const index = existingCanAnnounceProperties.indexOf( voicingCanSpeakProperty );
  assert && assert( index > -1, 'voicingNode.voicingCanSpeakProperty is not on the Utterance, was it not registered?' );
  utterance.voicingCanAnnounceProperties = existingCanAnnounceProperties.splice( index, 1 );
};

/**
 * Assign the Node's voicingVisibleProperty and visibleProperty to the Utterance so that the Utterance can only be
 * announced if the Node is visible and voicingVisible. This is LOCAL visibility and does not care about ancestors.
 * This should rarely be used, in general you should be registering an Utterance to a VoicingNode and its
 * voicingCanSpeakProperty.
 * @static
 */
Voicing.registerUtteranceToNode = ( utterance: Utterance, node: Node ) => {
  const existingCanAnnounceProperties = utterance.voicingCanAnnounceProperties;
  if ( !existingCanAnnounceProperties.includes( node.visibleProperty ) ) {
    utterance.voicingCanAnnounceProperties = utterance.voicingCanAnnounceProperties.concat( [ node.visibleProperty ] );
  }
  if ( !existingCanAnnounceProperties.includes( node.voicingVisibleProperty ) ) {
    utterance.voicingCanAnnounceProperties = utterance.voicingCanAnnounceProperties.concat( [ node.voicingVisibleProperty ] );
  }
};

/**
 * Remove a Node's voicingVisibleProperty and visibleProperty from the voicingCanAnnounceProperties of the Utterance.
 * @static
 */
Voicing.unregisterUtteranceToNode = ( utterance: Utterance, node: Node ) => {
  const existingCanAnnounceProperties = utterance.voicingCanAnnounceProperties;
  assert && assert( existingCanAnnounceProperties.includes( node.visibleProperty ) && existingCanAnnounceProperties.includes( node.voicingVisibleProperty ),
    'visibleProperty and voicingVisibleProperty were not on the Utterance, was it not registered to the node?' );

  const visiblePropertyIndex = existingCanAnnounceProperties.indexOf( node.visibleProperty );
  const withoutVisibleProperty = existingCanAnnounceProperties.splice( visiblePropertyIndex, 1 );

  const voicingVisiblePropertyIndex = withoutVisibleProperty.indexOf( node.voicingVisibleProperty );
  const withoutBothProperties = existingCanAnnounceProperties.splice( voicingVisiblePropertyIndex, 1 );

  utterance.voicingCanAnnounceProperties = withoutBothProperties;
};

/**
 * A basic behavior function for Voicing that sets both the ParallelDOM.accessibleName and Voicing.voicingNameResponse.
 * By using a behavior function, we can ensure that the accessibleName and voicingNameResponse are always in sync when you
 * mutate the accessibleName.
 *
 * For example:
 *   myNode.accessibleNameBehavior = Voicing.BASIC_ACCESSIBLE_NAME_BEHAVIOR;
 *   myNode.accessibleName = 'My Node'; // This will also set myNode.voicingNameResponse to 'My Node'
 */
Voicing.BASIC_ACCESSIBLE_NAME_BEHAVIOR = ( node: Node, options: ParallelDOMOptions, accessibleName: PDOMValueType ): ParallelDOMOptions => {
  assert && assert( isVoicing( node ), 'Node must be a VoicingNode to use Voicing.BASIC_ACCESSIBLE_NAME_BEHAVIOR' );
  const voicingNode = node as VoicingNode;

  // Create the basic accessible name options - the behavior function will use these options to mutate the Node.
  options = ParallelDOM.BASIC_ACCESSIBLE_NAME_BEHAVIOR( voicingNode, options, accessibleName );
  voicingNode.applyDefaultNameResponse( accessibleName );

  return options;
};

/**
 * A basic behavior function for Voicing that sets the both the ParallelDOM.accessibleHelpText and Voicing.voicingHintResponse.
 * By using a behavior function, we can ensure that the accessibleHelpText and voicingHintResponse are always in sync when you
 * mutate the accessibleHelpText.
 *
 * For example:
 *   myNode.accessibleHelpTextBehavior = Voicing.BASIC_HELP_TEXT_BEHAVIOR;
 *   myNode.accessibleHelpText = 'This is a helpful hint'; // This will also set myNode.voicingHintResponse to 'This is a helpful hint'
 */
Voicing.BASIC_HELP_TEXT_BEHAVIOR = ( node: Node, options: ParallelDOMOptions, accessibleHelpText: PDOMValueType ): ParallelDOMOptions => {
  assert && assert( isVoicing( node ), 'Node must be a VoicingNode to use Voicing.BASIC_HELP_TEXT_BEHAVIOR' );
  const voicingNode = node as VoicingNode;

  // Create the basic help text options - the behavior function will use these options to mutate the Node.
  options = ParallelDOM.HELP_TEXT_AFTER_CONTENT( voicingNode, options, accessibleHelpText );

  // If the voicingNameResponse has not been set manually, it will take the same value
  // as the accessibleName.
  voicingNode.applyDefaultHintResponse( accessibleHelpText );

  return options;
};

export type VoicingNode = Node & TVoicing;

export function isVoicing( something: IntentionalAny ): something is VoicingNode {
  return something instanceof Node && ( something as VoicingNode )._isVoicing;
}

scenery.register( 'Voicing', Voicing );
export default Voicing;