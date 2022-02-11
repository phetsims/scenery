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
import ResponsePacket, { ResponsePacketOptions } from '../../../../utterance-queue/js/ResponsePacket.js';
import ResponsePatternCollection from '../../../../utterance-queue/js/ResponsePatternCollection.js';
import Utterance from '../../../../utterance-queue/js/Utterance.js';
import UtteranceQueue from '../../../../utterance-queue/js/UtteranceQueue.js';
import { InteractiveHighlighting, Node, NodeOptions, scenery, SceneryListenerFunction, voicingUtteranceQueue } from '../../imports.js';
import optionize from '../../../../phet-core/js/optionize.js';
import Constructor from '../../../../phet-core/js/Constructor.js';
import { TAlertableDef } from '../../../../utterance-queue/js/AlertableDef.js';
import IntentionalAny from '../../../../phet-core/js/IntentionalAny.js';

// options that are supported by Voicing.js. Added to mutator keys so that Voicing properties can be set with mutate.
const VOICING_OPTION_KEYS = [
  'voicingNameResponse',
  'voicingObjectResponse',
  'voicingContextResponse',
  'voicingHintResponse',
  'voicingCreateNameResponse',
  'voicingCreateObjectResponse',
  'voicingCreateContextResponse',
  'voicingCreateHintResponse',
  'voicingUtteranceQueue',
  'voicingResponsePatternCollection',
  'voicingIgnoreVoicingManagerProperties',
  'voicingFocusListener'
];

type ResponseCreator = () => string | null;

type VoicingSelfOptions = {
  voicingNameResponse?: string | null,
  voicingObjectResponse?: string | null,
  voicingContextResponse?: string | null,
  voicingHintResponse?: string | null,
  voicingCreateNameResponse?: ResponseCreator | null,
  voicingCreateObjectResponse?: ResponseCreator | null,
  voicingCreateContextResponse?: ResponseCreator | null,
  voicingCreateHintResponse?: ResponseCreator | null,
  voicingUtteranceQueue?: UtteranceQueue,
  voicingResponsePatternCollection?: ResponsePatternCollection,
  voicingIgnoreVoicingManagerProperties?: boolean,
  voicingFocusListener?: SceneryListenerFunction | null
};

type VoicingOptions = VoicingSelfOptions & NodeOptions;

type ResponseOptions = {
  // The utterance to use if you want this response to be more controlled in the UtteranceQueue.
  utterance?: Utterance | null;
} & ResponsePacketOptions;

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

    // Instead of setting the name response as a string with voicingNameResponse, you can alternately pass in a
    // function that will be called to populate the  response for the response being spoken. If provided, this will
    // overwrite the current value of voicingNameResponse when speaking.
    _voicingCreateNameResponse!: ResponseCreator | null;

    // Instead of setting the object response as a string with voicingObjectResponse, you can alternately pass in a
    // function that will be called to populate the  response for the response being spoken. If provided, this will
    // overwrite the current value of voicingObjectResponse when speaking.
    _voicingCreateObjectResponse!: ResponseCreator | null;

    // Instead of setting the context response as a string with voicingContextResponse, you can alternately pass in a
    // function that will be called to populate the  response for the response being spoken. If provided, this will
    // overwrite the current value of voicingContextResponse when speaking.
    _voicingCreateContextResponse!: ResponseCreator | null;

    // Instead of setting the hint response as a string with voicingHintResponse, you can alternately pass in a
    // function that will be called to populate the  response for the response being spoken. If provided, this will
    // overwrite the current value of voicingHintResponse when speaking.
    _voicingCreateHintResponse!: ResponseCreator | null;

    // The utteranceQueue that responses for this Node will be spoken through.
    // By default (null), it will go through the singleton voicingUtteranceQueue, but you may need separate
    // UtteranceQueues for different areas of content in your application. For example, Voicing and
    // the default voicingUtteranceQueue may be disabled, but you could still want some speech to come through
    // while user is changing preferences or other settings.
    _voicingUtteranceQueue!: UtteranceQueue | null;

    // Called when this node is focused.
    _voicingFocusListener!: SceneryListenerFunction | null;

    // Input listener that speaks content on focus. This is the only input listener added
    // by Voicing, but it is the one that is consistent for all Voicing nodes. On focus, speak the name, object
    // response, and interaction hint.
    public _speakContentOnFocusListener!: { focus: SceneryListenerFunction };

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
      super.initialize && super.initialize();

      this._voicingResponsePacket = new ResponsePacket();
      this._voicingCreateNameResponse = null;
      this._voicingCreateObjectResponse = null;
      this._voicingCreateContextResponse = null;
      this._voicingCreateHintResponse = null;
      this._voicingUtteranceQueue = null;
      this._voicingFocusListener = this.defaultFocusListener;

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
    voicingSpeakFullResponse( providedOptions?: ResponseOptions ): void {

      // options are passed along to collectAndSpeakResponse, see that function for additional options
      const options = optionize<ResponseOptions, {}, ResponseOptions>( {}, providedOptions );

      // Lazily formulate strings only as needed
      if ( !options.hasOwnProperty( 'nameResponse' ) ) {
        options.nameResponse = this._getNameResponseToSpeak();
      }
      if ( !options.hasOwnProperty( 'objectResponse' ) ) {
        options.objectResponse = this._getObjectResponseToSpeak();
      }
      if ( !options.hasOwnProperty( 'contextResponse' ) ) {
        options.contextResponse = this._getContextResponseToSpeak();
      }
      if ( !options.hasOwnProperty( 'hintResponse' ) ) {
        options.hintResponse = this._getHintResponseToSpeak();
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
    voicingSpeakResponse( providedOptions?: ResponseOptions ): void {

      // options are passed along to collectAndSpeakResponse, see that function for additional options
      const options = optionize<ResponseOptions, {}, ResponseOptions>( {
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
    voicingSpeakNameResponse( providedOptions?: ResponseOptions ): void {

      // options are passed along to collectAndSpeakResponse, see that function for additional options
      const options = optionize<ResponseOptions>( {}, providedOptions );

      // Lazily formulate strings only as needed
      if ( !options.hasOwnProperty( 'nameResponse' ) ) {
        options.nameResponse = this._getNameResponseToSpeak();
      }

      this.collectAndSpeakResponse( options );
    }

    /**
     * By default, speak the object response. But accepts all other responses through options. Respects responseCollector
     * Properties, so the name response may not be spoken if responseCollector.objectResponseEnabledProperty is false.
     */
    voicingSpeakObjectResponse( providedOptions?: ResponseOptions ): void {

      // options are passed along to collectAndSpeakResponse, see that function for additional options
      const options = optionize<ResponseOptions>( {}, providedOptions );

      // Lazily formulate strings only as needed
      if ( !options.hasOwnProperty( 'objectResponse' ) ) {
        options.objectResponse = this._getObjectResponseToSpeak();
      }

      this.collectAndSpeakResponse( options );
    }

    /**
     * By default, speak the context response. But accepts all other responses through options. Respects
     * responseCollector Properties, so the name response may not be spoken if
     * responseCollector.contextResponseEnabledProperty is false.
     */
    voicingSpeakContextResponse( providedOptions?: ResponseOptions ): void {

      // options are passed along to collectAndSpeakResponse, see that function for additional options
      const options = optionize<ResponseOptions>( {}, providedOptions );

      // Lazily formulate strings only as needed
      if ( !options.hasOwnProperty( 'contextResponse' ) ) {
        options.contextResponse = this._getContextResponseToSpeak();
      }

      this.collectAndSpeakResponse( options );
    }

    /**
     * By default, speak the hint response. But accepts all other responses through options. Respects
     * responseCollector Properties, so the hint response may not be spoken if
     * responseCollector.hintResponseEnabledProperty is false.
     */
    voicingSpeakHintResponse( providedOptions?: ResponseOptions ): void {

      // options are passed along to collectAndSpeakResponse, see that function for additional options
      const options = optionize<ResponseOptions>( {}, providedOptions );

      // Lazily formulate strings only as needed
      if ( !options.hasOwnProperty( 'hintResponse' ) ) {
        options.hintResponse = this._getHintResponseToSpeak();
      }

      this.collectAndSpeakResponse( options );
    }

    /**
     * Collect responses with the responseCollector and speak the output with an UtteranceQueue.
     *
     * @protected
     */
    collectAndSpeakResponse( providedOptions?: ResponseOptions ): void {
      const options = optionize<ResponseOptions>( {
        ignoreProperties: this._voicingResponsePacket.ignoreProperties,
        responsePatternCollection: this._voicingResponsePacket.responsePatternCollection,
        utterance: null
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

    _getNameResponseToSpeak(): string | null {
      if ( this._voicingCreateNameResponse ) {
        this._voicingResponsePacket.nameResponse = this._voicingCreateNameResponse();
      }
      return this._voicingResponsePacket.nameResponse;
    }

    _getObjectResponseToSpeak(): string | null {
      if ( this._voicingCreateObjectResponse ) {
        this._voicingResponsePacket.objectResponse = this._voicingCreateObjectResponse();
      }
      return this._voicingResponsePacket.objectResponse;
    }

    _getContextResponseToSpeak(): string | null {
      if ( this._voicingCreateContextResponse ) {
        this._voicingResponsePacket.contextResponse = this._voicingCreateContextResponse();
      }
      return this._voicingResponsePacket.contextResponse;
    }

    _getHintResponseToSpeak(): string | null {
      if ( this._voicingCreateHintResponse ) {
        this._voicingResponsePacket.hintResponse = this._voicingCreateHintResponse();
      }
      return this._voicingResponsePacket.hintResponse;
    }

    /**
     * Sets the voicingNameResponse for this Node. This is usually the label of the element and is spoken
     * when the object receives input. When requesting speech, this will only be spoken if
     * responseCollector.nameResponsesEnabledProperty is set to true.
     */
    setVoicingNameResponse( response: string | null ): void {
      this._voicingResponsePacket.nameResponse = response;
    }

    set voicingNameResponse( response: string | null ) { this.setVoicingNameResponse( response ); }

    /**
     * Get the voicingNameResponse for this Node.
     */
    getVoicingNameResponse(): string | null {
      return this._voicingResponsePacket.nameResponse;
    }

    get voicingNameResponse(): string | null { return this.getVoicingNameResponse(); }

    /**
     * Set a function used to create the name response for this Node. Note that if using this setter, it will overwrite
     * the current value of voicingNameResponse when speaking.
     */
    setVoicingCreateNameResponse( responseCreator: ResponseCreator | null ) {
      this._voicingCreateNameResponse = responseCreator;
    }

    set voicingCreateNameResponse( responseCreator: ResponseCreator | null ) { this.setVoicingCreateNameResponse( responseCreator ); }

    /**
     * Gets the name-response creator function for this Node.
     */
    getVoicingCreateNameResponse(): ResponseCreator | null {
      return this._voicingCreateNameResponse;
    }

    get voicingCreateNameResponse(): ResponseCreator | null { return this.getVoicingCreateNameResponse(); }

    /**
     * Set the object response for this Node. This is usually the state information associated with this Node, such
     * as its current input value. When requesting speech, this will only be heard when
     * responseCollector.objectResponsesEnabledProperty is set to true.
     */
    setVoicingObjectResponse( response: string | null ) {
      this._voicingResponsePacket.objectResponse = response;
    }

    set voicingObjectResponse( response: string | null ) { this.setVoicingObjectResponse( response ); }

    /**
     * Gets the object response for this Node.
     */
    getVoicingObjectResponse(): string | null {
      return this._voicingResponsePacket.objectResponse;
    }

    get voicingObjectResponse(): string | null { return this.getVoicingObjectResponse(); }

    /**
     * Set a function used to create the object response for this Node. Note that if using this setter, it will overwrite
     * the current value of voicingObjectResponse when speaking.
     */
    setVoicingCreateObjectResponse( responseCreator: ResponseCreator | null ) {
      this._voicingCreateObjectResponse = responseCreator;
    }

    set voicingCreateObjectResponse( responseCreator: ResponseCreator | null ) { this.setVoicingCreateObjectResponse( responseCreator ); }

    /**
     * Gets the object-response creator function for this Node.
     */
    getVoicingCreateObjectResponse(): ResponseCreator | null {
      return this._voicingCreateObjectResponse;
    }

    get voicingCreateObjectResponse(): ResponseCreator | null { return this.getVoicingCreateObjectResponse(); }

    /**
     * Set the context response for this Node. This is usually the content that describes what has happened in
     * the surrounding application in response to interaction with this Node. When requesting speech, this will
     * only be heard if responseCollector.contextResponsesEnabledProperty is set to true.
     */
    setVoicingContextResponse( response: string | null ) {
      this._voicingResponsePacket.contextResponse = response;
    }

    set voicingContextResponse( response: string | null ) { this.setVoicingContextResponse( response ); }

    /**
     * Gets the context response for this Node.
     */
    getVoicingContextResponse(): string | null {
      return this._voicingResponsePacket.contextResponse;
    }

    get voicingContextResponse(): string | null { return this.getVoicingContextResponse(); }

    /**
     * Set a function used to create the context response for this Node. Note that if using this setter, it will overwrite
     * the current value of voicingContextResponse when speaking.
     */
    setVoicingCreateContextResponse( responseCreator: ResponseCreator | null ) {
      this._voicingCreateContextResponse = responseCreator;
    }

    set voicingCreateContextResponse( responseCreator: ResponseCreator | null ) { this.setVoicingCreateContextResponse( responseCreator ); }

    /**
     * Gets the context-response creator function for this Node.
     */
    getVoicingCreateContextResponse(): ResponseCreator | null {
      return this._voicingCreateContextResponse;
    }

    get voicingCreateContextResponse(): ResponseCreator | null { return this.getVoicingCreateContextResponse(); }

    /**
     * Sets the hint response for this Node. This is usually a response that describes how to interact with this Node.
     * When requesting speech, this will only be spoken when responseCollector.hintResponsesEnabledProperty is set to
     * true.
     */
    setVoicingHintResponse( response: string | null ) {
      this._voicingResponsePacket.hintResponse = response;
    }

    set voicingHintResponse( response: string | null ) { this.setVoicingHintResponse( response ); }

    /**
     * Gets the hint response for this Node.
     */
    getVoicingHintResponse(): string | null {
      return this._voicingResponsePacket.hintResponse;
    }

    get voicingHintResponse(): string | null { return this.getVoicingHintResponse(); }

    /**
     * Set a function used to create the hint response for this Node. Note that if using this setter, it will overwrite
     * the current value of voicingHintResponse when speaking.
     */
    setVoicingCreateHintResponse( responseCreator: ResponseCreator | null ) {
      this._voicingCreateHintResponse = responseCreator;
    }

    set voicingCreateHintResponse( responseCreator: ResponseCreator | null ) { this.setVoicingCreateHintResponse( responseCreator ); }

    /**
     * Gets the hint-response creator function for this Node.
     */
    getVoicingCreateHintResponse(): ResponseCreator | null {
      return this._voicingCreateHintResponse;
    }

    get voicingCreateHintResponse(): ResponseCreator | null { return this.getVoicingCreateHintResponse(); }

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
    setVoicingFocusListener( focusListener: SceneryListenerFunction | null ) {
      this._voicingFocusListener = focusListener;
    }

    set voicingFocusListener( focusListener: SceneryListenerFunction | null ) { this.setVoicingFocusListener( focusListener ); }

    /**
     * Gets the utteranceQueue through which voicing associated with this Node will be spoken.
     */
    getVoicingFocusListener(): SceneryListenerFunction | null {
      return this._voicingFocusListener;
    }

    get voicingFocusListener(): SceneryListenerFunction | null { return this.getVoicingFocusListener(); }


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

// @public
Voicing.VOICING_OPTION_KEYS = VOICING_OPTION_KEYS;

scenery.register( 'Voicing', Voicing );
export default Voicing;
export type { VoicingOptions };
