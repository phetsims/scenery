// Copyright 2021-2022, University of Colorado Boulder

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

import inheritance from '../../../../phet-core/js/inheritance.js';
import responseCollector from '../../../../utterance-queue/js/responseCollector.js';
import ResponsePacket, { ResponsePacketOptions, VoicingResponse } from '../../../../utterance-queue/js/ResponsePacket.js';
import ResponsePatternCollection from '../../../../utterance-queue/js/ResponsePatternCollection.js';
import Utterance from '../../../../utterance-queue/js/Utterance.js';
import UtteranceQueue from '../../../../utterance-queue/js/UtteranceQueue.js';
import { InteractiveHighlighting, InteractiveHighlightingOptions, Node, scenery, SceneryListenerFunction, voicingUtteranceQueue } from '../../imports.js';
import optionize from '../../../../phet-core/js/optionize.js';
import Constructor from '../../../../phet-core/js/types/Constructor.js';
import { TAlertableDef } from '../../../../utterance-queue/js/AlertableDef.js';
import IntentionalAny from '../../../../phet-core/js/IntentionalAny.js';

// options that are supported by Voicing.js. Added to mutator keys so that Voicing properties can be set with mutate.
const VOICING_OPTION_KEYS = [
  'voicingNameResponse',
  'voicingObjectResponse',
  'voicingContextResponse',
  'voicingHintResponse',
  'voicingUtterance',
  'voicingUtteranceQueue',
  'voicingResponsePatternCollection',
  'voicingIgnoreVoicingManagerProperties',
  'voicingFocusListener'
];

type VoicingSelfOptions = {

  // see ResponsePacket.nameResponse
  voicingNameResponse?: VoicingResponse,

  // see ResponsePacket.objectResponse
  voicingObjectResponse?: VoicingResponse,

  // see ResponsePacket.contextResponse
  voicingContextResponse?: VoicingResponse,

  // see ResponsePacket.hintResponse
  voicingHintResponse?: VoicingResponse,

  // see ResponsePacket.responsePatternCollection
  voicingResponsePatternCollection?: ResponsePatternCollection,

  // see ResponsePacket.ignoreProperties
  voicingIgnoreVoicingManagerProperties?: boolean,

  // Called when this Node is focused to speak voicing responses on focus. See Voicing.defaultFocusListener for default
  // listener.
  voicingFocusListener?: SceneryListenerFunction<FocusEvent> | null

  // By default use voicingUtteranceQueue to speak responses, but you can also specify another utteranceQueue here.
  voicingUtteranceQueue?: UtteranceQueue,

  // The utterance to use if you want this response to be more controlled in the UtteranceQueue. This Utterance will be
  // used by all responses spoken by this class.
  voicingUtterance?: Utterance;
};

type VoicingOptions = VoicingSelfOptions & InteractiveHighlightingOptions;

type SpeakingOptions = {
  utterance?: VoicingSelfOptions['voicingUtterance']
} & {

  // In speaking options, we don't allow a ResponseCreator function, but just a string|null. The `undefined` is to
  // match on the properties because they are optional (marked with `?`)
  [PropertyName in keyof ResponsePacketOptions]: ResponsePacketOptions[PropertyName] extends ( VoicingResponse | undefined ) ?
                                                 ( string | null ) :
                                                 ResponsePacketOptions[PropertyName];
}

/**
 * @param Type
 * @param optionsArgPosition - zero-indexed number that the options argument is provided at
 */
const Voicing = <SuperType extends Constructor>( Type: SuperType, optionsArgPosition: number ) => {

  assert && assert( _.includes( inheritance( Type ), Node ), 'Only Node subtypes should compose Voicing' );

  // Unfortunately, nothing can be private or protected in this class, see https://github.com/phetsims/scenery/issues/1340#issuecomment-1020692592
  const VoicingClass = class extends InteractiveHighlighting( Type, optionsArgPosition ) {

    // ResponsePacket that holds all the supported responses to be Voiced
    _voicingResponsePacket!: ResponsePacket;

    // The utterance that all responses are spoken through.
    _voicingUtterance!: Utterance;

    // The utteranceQueue that responses for this Node will be spoken through.
    // By default (null), it will go through the singleton voicingUtteranceQueue, but you may need separate
    // UtteranceQueues for different areas of content in your application. For example, Voicing and
    // the default voicingUtteranceQueue may be disabled, but you could still want some speech to come through
    // while user is changing preferences or other settings.
    _voicingUtteranceQueue!: UtteranceQueue | null;

    // Called when this node is focused.
    _voicingFocusListener!: SceneryListenerFunction<FocusEvent> | null;

    // Input listener that speaks content on focus. This is the only input listener added
    // by Voicing, but it is the one that is consistent for all Voicing nodes. On focus, speak the name, object
    // response, and interaction hint.
    _speakContentOnFocusListener!: { focus: SceneryListenerFunction<FocusEvent> };

    constructor( ...args: IntentionalAny[] ) {

      const providedOptions = ( args[ optionsArgPosition ] || {} ) as VoicingOptions;

      const voicingOptions = _.pick( providedOptions, VOICING_OPTION_KEYS );
      args[ optionsArgPosition ] = _.omit( providedOptions, VOICING_OPTION_KEYS );

      super( ...args );

      // We only want to call this method, not any subtype implementation
      VoicingClass.prototype.initialize.call( this );

      ( this as unknown as Node ).mutate( voicingOptions );
    }

    // Separate from the constructor to support cases where Voicing is used in Poolable Nodes.
    // ...args: IntentionalAny[] because things like RichTextLink need to provide arguments to initialize, and TS complains
    // otherwise
    initialize( ...args: IntentionalAny[] ): this {

      // @ts-ignore
      super.initialize && super.initialize( args );

      this._voicingResponsePacket = new ResponsePacket();
      this._voicingFocusListener = this.defaultFocusListener;
      this._voicingUtterance = new Utterance();

      this._speakContentOnFocusListener = {
        focus: event => {
          this._voicingFocusListener && this._voicingFocusListener( event );
        }
      };
      ( this as unknown as Node ).addInputListener( this._speakContentOnFocusListener );

      return this;
    }

    /**
     * Speak all responses assigned to this Node. Options allow you to override a responses for this particular
     * speech request. Each response is only spoken if the associated Property of responseCollector is true. If
     * all are Properties are false, nothing will be spoken.
     */
    voicingSpeakFullResponse( providedOptions?: SpeakingOptions ): void {

      // options are passed along to collectAndSpeakResponse, see that function for additional options
      const options = optionize<SpeakingOptions, {}, SpeakingOptions>( {}, providedOptions );

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
    voicingSpeakResponse( providedOptions?: SpeakingOptions ): void {

      // options are passed along to collectAndSpeakResponse, see that function for additional options
      const options = optionize<SpeakingOptions, {}, SpeakingOptions>( {
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
    voicingSpeakNameResponse( providedOptions?: SpeakingOptions ): void {

      // options are passed along to collectAndSpeakResponse, see that function for additional options
      const options = optionize<SpeakingOptions, {}, SpeakingOptions>( {}, providedOptions );

      // Lazily formulate strings only as needed
      if ( !options.hasOwnProperty( 'nameResponse' ) ) {
        options.nameResponse = this._voicingResponsePacket.nameResponse;
      }

      this.collectAndSpeakResponse( options );
    }

    /**
     * By default, speak the object response. But accepts all other responses through options. Respects responseCollector
     * Properties, so the name response may not be spoken if responseCollector.objectResponseEnabledProperty is false.
     */
    voicingSpeakObjectResponse( providedOptions?: SpeakingOptions ): void {

      // options are passed along to collectAndSpeakResponse, see that function for additional options
      const options = optionize<SpeakingOptions, {}, SpeakingOptions>( {}, providedOptions );

      // Lazily formulate strings only as needed
      if ( !options.hasOwnProperty( 'objectResponse' ) ) {
        options.objectResponse = this._voicingResponsePacket.objectResponse;
      }

      this.collectAndSpeakResponse( options );
    }

    /**
     * By default, speak the context response. But accepts all other responses through options. Respects
     * responseCollector Properties, so the name response may not be spoken if
     * responseCollector.contextResponseEnabledProperty is false.
     */
    voicingSpeakContextResponse( providedOptions?: SpeakingOptions ): void {

      // options are passed along to collectAndSpeakResponse, see that function for additional options
      const options = optionize<SpeakingOptions, {}, SpeakingOptions>( {}, providedOptions );

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
    voicingSpeakHintResponse( providedOptions?: SpeakingOptions ): void {

      // options are passed along to collectAndSpeakResponse, see that function for additional options
      const options = optionize<SpeakingOptions, {}, SpeakingOptions>( {}, providedOptions );

      // Lazily formulate strings only as needed
      if ( !options.hasOwnProperty( 'hintResponse' ) ) {
        options.hintResponse = this._voicingResponsePacket.hintResponse;
      }

      this.collectAndSpeakResponse( options );
    }

    /**
     * Collect responses with the responseCollector and speak the output with an UtteranceQueue.
     *
     * @protected
     */
    collectAndSpeakResponse( providedOptions?: SpeakingOptions ): void {
      const options = optionize<SpeakingOptions, {}, SpeakingOptions>( {
        ignoreProperties: this._voicingResponsePacket.ignoreProperties,
        responsePatternCollection: this._voicingResponsePacket.responsePatternCollection,
        utterance: this.voicingUtterance
      }, providedOptions );

      let response: TAlertableDef = responseCollector.collectResponses( options ); // eslint-disable-line no-undef

      if ( options.utterance ) {
        options.utterance.alert = response;
        response = options.utterance;
      }
      this.speakContent( response );
    }

    /**
     * Use the provided function to create content to speak in response to input. The content is then added to the
     * back of the voicing UtteranceQueue.
     * @protected
     */
    speakContent( content: TAlertableDef | null ): void { // eslint-disable-line no-undef

      // don't send to utteranceQueue if response is empty
      if ( content ) {
        const utteranceQueue = this.voicingUtteranceQueue || voicingUtteranceQueue;
        utteranceQueue.addToBack( content );
      }
    }

    /**
     * Sets the voicingNameResponse for this Node. This is usually the label of the element and is spoken
     * when the object receives input. When requesting speech, this will only be spoken if
     * responseCollector.nameResponsesEnabledProperty is set to true.
     */
    setVoicingNameResponse( response: VoicingResponse ): void {
      this._voicingResponsePacket.nameResponse = response;
    }

    set voicingNameResponse( response: VoicingResponse ) { this.setVoicingNameResponse( response ); }

    /**
     * Get the voicingNameResponse for this Node.
     */
    getVoicingNameResponse(): VoicingResponse {
      return this._voicingResponsePacket.nameResponse;
    }

    get voicingNameResponse(): VoicingResponse { return this.getVoicingNameResponse(); }

    /**
     * Set the object response for this Node. This is usually the state information associated with this Node, such
     * as its current input value. When requesting speech, this will only be heard when
     * responseCollector.objectResponsesEnabledProperty is set to true.
     */
    setVoicingObjectResponse( response: VoicingResponse ) {
      this._voicingResponsePacket.objectResponse = response;
    }

    set voicingObjectResponse( response: VoicingResponse ) { this.setVoicingObjectResponse( response ); }

    /**
     * Gets the object response for this Node.
     */
    getVoicingObjectResponse(): string | null {
      return this._voicingResponsePacket.objectResponse;
    }

    get voicingObjectResponse(): string | null { return this.getVoicingObjectResponse(); }

    /**
     * Set the context response for this Node. This is usually the content that describes what has happened in
     * the surrounding application in response to interaction with this Node. When requesting speech, this will
     * only be heard if responseCollector.contextResponsesEnabledProperty is set to true.
     */
    setVoicingContextResponse( response: VoicingResponse ) {
      this._voicingResponsePacket.contextResponse = response;
    }

    set voicingContextResponse( response: VoicingResponse ) { this.setVoicingContextResponse( response ); }

    /**
     * Gets the context response for this Node.
     */
    getVoicingContextResponse(): string | null {
      return this._voicingResponsePacket.contextResponse;
    }

    get voicingContextResponse(): string | null { return this.getVoicingContextResponse(); }

    /**
     * Sets the hint response for this Node. This is usually a response that describes how to interact with this Node.
     * When requesting speech, this will only be spoken when responseCollector.hintResponsesEnabledProperty is set to
     * true.
     */
    setVoicingHintResponse( response: VoicingResponse ) {
      this._voicingResponsePacket.hintResponse = response;
    }

    set voicingHintResponse( response: VoicingResponse ) { this.setVoicingHintResponse( response ); }

    /**
     * Gets the hint response for this Node.
     */
    getVoicingHintResponse(): string | null {
      return this._voicingResponsePacket.hintResponse;
    }

    get voicingHintResponse(): string | null { return this.getVoicingHintResponse(); }

    /**
     * Set whether or not all responses for this Node will ignore the Properties of responseCollector. If false,
     * all responses will be spoken regardless of responseCollector Properties, which are generally set in user
     * preferences.
     */
    setVoicingIgnoreVoicingManagerProperties( ignoreProperties: boolean ) {
      this._voicingResponsePacket.ignoreProperties = ignoreProperties;
    }

    set voicingIgnoreVoicingManagerProperties( ignoreProperties: boolean ) { this.setVoicingIgnoreVoicingManagerProperties( ignoreProperties ); }

    /**
     * Get whether or not responses are ignoring responseCollector Properties.
     */
    getVoicingIgnoreVoicingManagerProperties(): boolean {
      return this._voicingResponsePacket.ignoreProperties;
    }

    get voicingIgnoreVoicingManagerProperties(): boolean { return this.getVoicingIgnoreVoicingManagerProperties(); }

    /**
     * Sets the collection of patterns to use for voicing responses, controlling the order, punctuation, and
     * additional content for each combination of response. See ResponsePatternCollection.js if you wish to use
     * a collection of string patterns that are not the default.
     */
    setVoicingResponsePatternCollection( patterns: ResponsePatternCollection ) {
      assert && assert( patterns instanceof ResponsePatternCollection );

      this._voicingResponsePacket.responsePatternCollection = patterns;
    }

    set voicingResponsePatternCollection( patterns: ResponsePatternCollection ) { this.setVoicingResponsePatternCollection( patterns ); }

    /**
     * Get the ResponsePatternCollection object that this Voicing Node is using to collect responses.
     */
    getVoicingResponsePatternCollection(): ResponsePatternCollection {
      return this._voicingResponsePacket.responsePatternCollection;
    }

    get voicingResponsePatternCollection(): ResponsePatternCollection { return this.getVoicingResponsePatternCollection(); }

    /**
     * Sets the utterance through which voicing associated with this Node will be spoken. By default on initialize,
     * one will be created, but a custom one can optionally be provided.
     */
    setVoicingUtterance( utterance: Utterance ) {
      this._voicingUtterance = utterance;
    }

    set voicingUtterance( utterance: Utterance ) { this.setVoicingUtterance( utterance ); }

    /**
     * Gets the utterance through which voicing associated with this Node will be spoken.
     */
    getVoicingUtterance(): Utterance {
      return this._voicingUtterance;
    }

    get voicingUtterance(): Utterance { return this.getVoicingUtterance(); }

    /**
     * Sets the utteranceQueue through which voicing associated with this Node will be spoken. By default,
     * the Display's voicingUtteranceQueue is used. But you can specify a different one if more complicated
     * management of voicing is necessary.
     */
    setVoicingUtteranceQueue( utteranceQueue: UtteranceQueue | null ) {
      this._voicingUtteranceQueue = utteranceQueue;
    }

    set voicingUtteranceQueue( utteranceQueue: UtteranceQueue | null ) { this.setVoicingUtteranceQueue( utteranceQueue ); }

    /**
     * Gets the utteranceQueue through which voicing associated with this Node will be spoken.
     */
    getVoicingUtteranceQueue(): UtteranceQueue | null {
      return this._voicingUtteranceQueue;
    }

    get voicingUtteranceQueue(): UtteranceQueue | null { return this.getVoicingUtteranceQueue(); }

    /**
     * Called whenever this Node is focused.
     */
    setVoicingFocusListener( focusListener: SceneryListenerFunction<FocusEvent> | null ) {
      this._voicingFocusListener = focusListener;
    }

    set voicingFocusListener( focusListener: SceneryListenerFunction<FocusEvent> | null ) { this.setVoicingFocusListener( focusListener ); }

    /**
     * Gets the utteranceQueue through which voicing associated with this Node will be spoken.
     */
    getVoicingFocusListener(): SceneryListenerFunction<FocusEvent> | null {
      return this._voicingFocusListener;
    }

    get voicingFocusListener(): SceneryListenerFunction<FocusEvent> | null { return this.getVoicingFocusListener(); }


    /**
     * The default focus listener attached to this Node during initialization.
     */
    defaultFocusListener(): void {
      this.voicingSpeakFullResponse( {
        contextResponse: null
      } );
    }

    /**
     * Whether or not a Node composes Voicing.
     */
    get isVoicing(): boolean {
      return true;
    }

    /**
     * Detaches references that ensure this components of this Trait are eligible for garbage collection.
     */
    dispose() {
      ( this as unknown as Node ).removeInputListener( this._speakContentOnFocusListener );

      super.dispose();
    }

    clean() {
      ( this as unknown as Node ).removeInputListener( this._speakContentOnFocusListener );

      // @ts-ignore
      super.clean && super.clean();
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
  VoicingClass.prototype._mutatorKeys = VOICING_OPTION_KEYS.concat( VoicingClass.prototype._mutatorKeys );
  assert && assert( VoicingClass.prototype._mutatorKeys.length === _.uniq( VoicingClass.prototype._mutatorKeys ).length, 'duplicate mutator keys in Voicing' );

  return VoicingClass;
};

Voicing.VOICING_OPTION_KEYS = VOICING_OPTION_KEYS;

scenery.register( 'Voicing', Voicing );
export default Voicing;
export type { VoicingOptions };